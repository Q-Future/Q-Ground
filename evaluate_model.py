import argparse
import os
import sys
import json
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.PixelLM import PixelLMForCausalLM

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

from utils_label import DISTORTION_TYPES, rgb2label 

import shutil


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="chaofengc/qg_model_multiscale")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    parser.add_argument("--seg_token_num", default=3, type=int)
    parser.add_argument("--image_feature_scale_num", default=2, type=int)

    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=448, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--pad_train_clip_images", action="store_true", default=False)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)
    parser.add_argument("--use_multi_scale_mm_projector", action="store_true", default=False)
    parser.add_argument("--mm_vision_select_layer", type=str, default='6,-11,-2')
    parser.add_argument("--preprocessor_config", default='./configs/preprocessor_448.json', type=str)
    parser.add_argument("--save_results_path", default=None, type=str)
    parser.add_argument("--not_use_quality_txt", action="store_true", default=False)

    parser.add_argument("--debug", action="store_true", default=False)

    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def load_model(args):
    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num*args.image_feature_scale_num)]
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    args.seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {
        "torch_dtype": torch_dtype,  
        "seg_token_num": args.seg_token_num, 
        "image_feature_scale_num": args.image_feature_scale_num, "pad_train_clip_images": args.pad_train_clip_images, 
        "resize_vision_tower": args.resize_vision_tower,
        "resize_vision_tower_size": args.resize_vision_tower_size,
        "vision_tower_for_mask": args.vision_tower_for_mask,
        "separate_mm_projector": args.separate_mm_projector,
        "use_multi_scale_mm_projector": args.use_multi_scale_mm_projector,
        "mm_vision_select_layer": args.mm_vision_select_layer,
    }

    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = PixelLMForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx,  **kwargs
        )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=False,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)
    
    model.eval()

    return model, tokenizer, args


@torch.no_grad()
def model_inference(input_str, input_image, model, tokenizer, args, return_all=False):

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower) if args.preprocessor_config == '' else CLIPImageProcessor.from_pretrained(args.preprocessor_config)
    transform = ResizeLongestSide(args.image_size)
    if args.pad_train_clip_images:
        transform_clip = ResizeLongestSide(clip_image_processor.size['shortest_edge'])

    # Model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = input_str
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    if isinstance(input_image, str):   
        image_np = cv2.imread(input_image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    elif isinstance(input_image, Image.Image):
        image_np = np.array(input_image.convert('RGB'))

    original_size_list = [image_np.shape[:2]]

    if args.pad_train_clip_images:
        image_clip = transform_clip.apply_image(image_np)
        clip_resize = image_clip.shape[:2]
        image_clip = preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), img_size=clip_image_processor.size['shortest_edge'])
        image_clip = image_clip.unsqueeze(0).cuda()
    else:
        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        clip_resize = image_clip.shape[-2:]

    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    clip_resize = [clip_resize]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    model.eval()
    output_ids, pred_masks, _, _ = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        clip_resize_list=clip_resize,
        original_size_list=original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer
    )

    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1].replace('</s>', '')

    output_masks = []
    for i, _pred_mask in enumerate(pred_masks):
        if _pred_mask.shape[0] == 0:
            continue
        for j, pred_mask in enumerate(_pred_mask):
            pred_mask = pred_mask.float().detach().cpu().numpy()
            pred_mask = pred_mask > 0
            output_masks.append(pred_mask)
    
    distortion_types = ['jitter', 'noise', 'overexposure', 'blur', 'low light']

    masks_dict = {}
    if len(output_masks) > 0:
        tmp_output_masks = output_masks.copy()
        for dt in distortion_types:
            if dt in text_output: 
                masks_dict[dt] = tmp_output_masks.pop(0)
    
    if return_all:

        if len(output_masks) > 0:
            output_imgs = []
            for mask in output_masks:
                tmp_img = mask[:, :, None] * np.array([255, 0, 0]) * 0.5 + image_np * 0.5
                output_imgs.append(tmp_img.astype(np.uint8))
            output_imgs = np.concatenate(output_imgs, axis=1)

            return text_output, Image.fromarray(output_imgs.astype(np.uint8))
        else:
            return text_output, None
    else:
        return masks_dict 


def intersect_and_union(pred_label, label, num_classes, ignore_index=None):
    if ignore_index is not None:
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

    # calculate intersection, i.e., True Positives for each class
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)

    # calculate predicted area and ground truth area, i.e., predicted pixels and GT pixels for each class
    area_pred = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    # calculate union, i.e., True Positives + False Positives + False Negatives for each class
    # Union = Area_Pred + Area_Label - Intersection
    area_union = area_pred + area_label - area_intersect

    return area_intersect, area_union, area_pred, area_label


def get_iou_acc(pred_results_path):

    test_info_file = './dataset/qground_dataset/qground_test.json'

    with open(test_info_file, 'r') as f:
        test_info = json.load(f)

    metric_all = []
    for test_idx, ti in tqdm(enumerate(test_info), total=len(test_info)):
        for ai in ti['ann_list']:
            gt_img_name = ai['segmentation_mask']

            gt_img_path = os.path.join('./dataset/qground/masks/', gt_img_name)
            pred_img_path = os.path.join(pred_results_path, os.path.basename(gt_img_name))
            
            gt_labels = rgb2label(np.array(Image.open(gt_img_path)))
            pred_labels = rgb2label(np.array(Image.open(pred_img_path)))

            # 5 distortion types + 1 background class 
            metric_all.append(torch.stack(intersect_and_union(torch.from_numpy(pred_labels).float(), torch.from_numpy(gt_labels).float(), num_classes=6)))

    # calculate global mIoU and mAcc
    # sum over all samples and ignore background class
    metric_total = torch.stack(metric_all).sum(dim=0)[:, 1:]    
    total_intersect, total_union, total_pred, total_label = metric_total.chunk(4, dim=0)

    # Class-wise IoU and mIoU
    iou = total_intersect / (total_union + 1e-6)
    miou = torch.mean(iou)

    # Class-wise Accuracy (Recall) and mAcc
    # Acc = TP / GT_Pixels
    acc = total_intersect / (total_label + 1e-6)
    macc = torch.mean(acc)
    
    # Global Accuracy (Overall pixel accuracy across all classes)
    # Total_Correct_Pixels / Total_Labeled_Pixels
    all_acc = total_intersect.sum() / (total_label.sum() + 1e-6)

    print(f"Class IoU: {iou}, Class Acc: {acc}")
    print(f"mIoU: {miou}, mAcc: {macc}, Global Acc: {all_acc}")


def get_results(args):

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model, tokenizer, args = load_model(args)

    img_dir = './dataset/qground_dataset/images/'
    test_info_file = './dataset/qground_dataset/qground_test.json'

    with open(test_info_file, 'r') as f:
        test_info = json.load(f)

    for test_idx, ti in tqdm(enumerate(test_info), total=len(test_info)):
        img_name = ti['image']
        img_path = os.path.join(img_dir, img_name)
        for ai in ti['ann_list']:
            quality_txt = ai['quality_description']
            gt_img_name = ai['segmentation_mask']

            if args.not_use_quality_txt:
                prompt = 'Please segment out distorted regions in the image.'
            else:
                prompt = f'{quality_txt} Please segment out distorted regions in the image.'
            masks = model_inference(prompt, img_path, model, tokenizer, args)

            # masks to rgb
            img_size = np.array(Image.open(img_path)).shape[:2]
            filled = np.zeros(img_size, dtype=np.uint8)
            mask_rgb = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            sorted_mask = dict(sorted(masks.items(), key=lambda item: item[1].sum()))

            for idx, (dt, mask) in enumerate(sorted_mask.items()):
                mask_rgb = mask_rgb * filled[:, :, None] + (1 - filled[:, :, None]) * mask[:, :, None] * np.array(DISTORTION_TYPES[dt])
                filled = np.logical_or(filled, mask)

            save_path = os.path.join(save_dir, os.path.basename(gt_img_name))
            Image.fromarray(mask_rgb.astype(np.uint8)).save(save_path)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    save_dir = args.save_results_path
    get_results(args)
    get_iou_acc(save_dir)