#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
#                          DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
#                          IMAGE_TOKEN_INDEX)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_encoder.custom_clip import _CLIPVisionModel
import torch.nn.functional as F

import math
from torch import Tensor


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # # Get output
        out = attn @ v

        out = self._recombine_heads(out)

        out = self.out_proj(out)

        return out


def safe_interpolate(x, **kwargs):
    return F.interpolate(x.float(), **kwargs).to(x)


class MultiScalePrj(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_layers=2, num_heads=8):
        super().__init__()

        self.q_proj = nn.Linear(1024, hidden_dim)
        self.kv_proj = nn.Linear(in_dim, hidden_dim)

        self.multihead_attn = Attention(hidden_dim, num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, query, img_features):

        query = self.q_proj(query)
        img_features = self.kv_proj(img_features)
       
        query = query + self.multihead_attn(query, img_features, img_features)
        query = query + self.mlp(query) 

        return self.out_proj(query) 


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        # import pdb;pdb.set_trace()
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        self.separate_mm_projector = config.separate_mm_projector
        if config.separate_mm_projector:
            hidden_size = config.hidden_size if config.mm_projector_hidden_dim == 1 else config.hidden_size*2
            out_size = config.hidden_size if config.mm_projector_out_dim == 1 else config.mm_hidden_size

            if config.use_multi_scale_mm_projector:
                self.mm_projector = MultiScalePrj(config.hidden_size)
            else:
                self.mm_projector = nn.Sequential(nn.Linear(config.mm_hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, config.hidden_size))

            self.out_mm_projector = nn.Sequential(nn.Linear(config.mm_hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, out_size))
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.use_multi_scale_mm_projector = model_args.use_multi_scale_mm_projector
        
        self.config.use_mm_proj = True
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        print(model_args)

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.mm_hidden_size = vision_tower.hidden_size

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, clip_resize_list, return_project=False):
        vision_tower = self.get_model().get_vision_tower()
        vit_attention_mask_for_llm = None
        if isinstance(vision_tower.vision_tower, _CLIPVisionModel):
            vit_attention_mask = torch.zeros_like(images[:, 0, :, :])
            for i, size in enumerate(clip_resize_list):
                vit_attention_mask[i, :size[0], :size[1]] = 1
            with torch.no_grad():
                patch_size = 14
                patch_num = vit_attention_mask.shape[-1] // patch_size
                vit_attention_mask = safe_interpolate(vit_attention_mask[:, None], size=(patch_num, patch_num), mode="nearest")[:, 0]
                vit_attention_mask_for_llm = safe_interpolate(vit_attention_mask[:, None], size=(16, 16), mode="nearest")[:, 0]
                vit_attention_mask_for_llm = vit_attention_mask_for_llm.flatten(1)
                flatten_vit_attention_mask = vit_attention_mask.flatten(1)
                flatten_vit_attention_mask = torch.cat((torch.ones(flatten_vit_attention_mask.shape[0], 1).to(flatten_vit_attention_mask), flatten_vit_attention_mask), dim=-1)
            image_features, pre_image_features = vision_tower(images, attention_mask=flatten_vit_attention_mask)
        else:
            image_features, pre_image_features = self.get_model().get_vision_tower()(images)

        # image_features = self.get_model().mm_projector(image_features)
        if return_project:
            # image_features = self.get_model().mm_projector(image_features)
            pre_image_features = [self.get_model().out_mm_projector(f) if self.get_model().separate_mm_projector else self.get_model().mm_projector(f) for f in pre_image_features]
            output_image_features = [self.get_model().out_mm_projector(image_features) if self.get_model().separate_mm_projector else self.get_model().mm_projector(image_features)]
            output_image_features.extend(pre_image_features)
            # use two scale features for decoder, same as PixelLM
            pre_image_features = output_image_features[-2:]

        return image_features, vit_attention_mask_for_llm, pre_image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, clip_resize_list
    ):
        vision_tower = self.get_vision_tower()
        vit_attention_mask = None
        # import pdb;pdb.set_trace()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (input_ids.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            return None, input_ids, attention_mask, past_key_values, None, labels

            # output_image_features, None, attention_mask, past_key_values, new_input_embeds, new_labels
        # import pdb;pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            assert False
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features, vit_attention_mask, pre_image_features = self.encode_images(images, clip_resize_list)
            attention_mask = torch.ones_like(input_ids).detach() if attention_mask is None else attention_mask
        # import pdb;pdb.set_trace()
        pre_image_features = [self.get_model().out_mm_projector(f) if self.get_model().separate_mm_projector else self.get_model().mm_projector(f) for f in pre_image_features]

        output_image_features = [self.get_model().out_mm_projector(image_features) if self.get_model().separate_mm_projector else self.get_model().mm_projector(image_features)]
        output_image_features.extend(pre_image_features)
        # output_image_features = output_image_features[1:]

        n, l, c = image_features.shape
        p_num = int(l ** 0.5)
        image_features = safe_interpolate(image_features.permute(0, 2, 1).view(n, c, p_num, p_num).float(), size=(16,16),  mode="bilinear",align_corners=False).to(image_features)
        image_features = image_features.flatten(-2).permute(0, 2, 1)

        if self.config.use_multi_scale_mm_projector:
            image_features = self.get_model().mm_projector(image_features, torch.cat(output_image_features, dim=1))
        else:
            image_features = self.get_model().mm_projector(image_features)
        
        # use two scale features for decoder, same as PixelLM
        output_image_features = output_image_features[-2:]
        
        vit_attention_mask = torch.ones_like(image_features[:,:,0]).detach() if vit_attention_mask is None else vit_attention_mask

        new_input_embeds = []
        new_attention_mask = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_new_attention_mask = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                cur_attention_mask = attention_mask[cur_image_idx]
                cur_vit_attention_mask = vit_attention_mask[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False):
                    assert False
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_attention_mask.append(cur_attention_mask[:image_token_start])
                    cur_new_attention_mask.append(cur_vit_attention_mask)
                    cur_new_attention_mask.append(cur_attention_mask[image_token_start + 1 : image_token_start + 2])
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_attention_mask.append(
                        cur_attention_mask[:image_token_start]
                    )
                    cur_new_attention_mask.append(
                        cur_vit_attention_mask
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                    cur_attention_mask = cur_attention_mask[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                    cur_attention_mask = cur_attention_mask[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                    cur_attention_mask = cur_attention_mask[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                cur_new_attention_mask.append(cur_attention_mask)
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_attention_mask = torch.cat(cur_new_attention_mask, dim=0).bool()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            new_attention_mask.append(cur_new_attention_mask)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds): #the image lengths are not the same
            assert False
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_attention_mask = torch.stack(new_attention_mask, dim=0)
            attention_mask = new_attention_mask
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None and attention_mask.shape[1] < new_input_embeds.shape[1]:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return output_image_features, None, attention_mask, past_key_values, new_input_embeds, new_labels

    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    def initialize_vision_tokenizer(self, model_args, num_new_tokens):
        # if model_args.mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        #     self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            # num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # self.resize_token_embeddings(len(tokenizer))

            # if num_new_tokens > 0:
            #     input_embeddings = self.get_input_embeddings().weight.data
            #     output_embeddings = self.get_output_embeddings().weight.data

            #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)
            #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)

            #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
            #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


def visualize_attn_mask(mask):
    import cv2
    import numpy as np
    mask = mask[0].squeeze().float()
    fg = mask >= 0
    mask_show = torch.zeros_like(mask)
    mask_show[fg] = 255
    mask_show = mask_show.cpu().numpy()
    cv2.imwrite('test.jpg', mask_show.astype(np.uint8))