import json
import os
import random 
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import transformers
# from dataset import clip_image_process
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
# from .conversation import get_default_conv_template

from .utils import (
    MR_SINGLE_ANSWER_LIST,
    MR_MULTI_ANSWER_LIST,
    ANSWER_LIST,
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
    EXPAND_LONG_QUESTION_LIST,
    EXPAND_QUESTION_LIST,
)
from transformers.image_utils import make_list_of_images, to_numpy_array, infer_channel_dimension_format
from transformers.image_transforms import convert_to_rgb


import sys 
sys.path.append(os.path.abspath("../"))
from utils_label import get_distortion_mask, get_all_distortion_masks


DISTORTION_SEG_LIST = [
    'distortion segmentation mask',
    'segmentation mask for distorted regions',
    'segmentation mask for identifying distorted regions',
    'segmentation mask for distortions in the image', 
    'segmentation mask to distortions', 
    'distorted regions in the image',
    'distortions in the image',
    'low quality regions in the image',
]

NO_DISTORTION_ANSWER = [
    "The image quality is good without distortions.",
    "No distortion found in the image.",
    "The image quality is good with little distortions.",
    "There is no obvious distortions in the image",
    "The image is clear without distortions.",
    "The image is clear with little distortions.",
]


class QGroundDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="QGround|train",
        explanatory=0.1,
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,
        use_quality_description=True,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        self.use_quality_description = use_quality_description

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.single_answer_list = MR_SINGLE_ANSWER_LIST
        self.multi_answer_list = MR_MULTI_ANSWER_LIST   
        self.seg_token_num = seg_token_num
        self.num_classes_per_question = num_classes_per_question
        
        self.masks_process_with_clip = masks_process_with_clip
        self.pad_train_clip_images = pad_train_clip_images
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge']) 
        
        if use_expand_question_list:
            self.long_question_list.extend(EXPAND_LONG_QUESTION_LIST)
            self.short_question_list.extend(EXPAND_QUESTION_LIST)
        

        print("___________self.single_answer_list:", self.single_answer_list)
        print("___________self.multi_answer_list:", self.multi_answer_list)
                
        reason_seg_data, split = reason_seg_data.split("|")
        json_file_name = './dataset/qground_dataset/qground_train.json'
       
        with open(json_file_name, 'r') as f:
            reason_file = json.load(f)
        
        self.reason_seg_data = reason_file

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor, decoder_image_size) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = decoder_image_size - h
        padw = decoder_image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __getitem__(self, idx):

        self.long_question_list = [q.replace('segmentation mask', random.choice(DISTORTION_SEG_LIST)) for q in self.long_question_list]

        #
        idx = random.randint(0, len(self.reason_seg_data) - 1)
        image_info = self.reason_seg_data[idx]

        image_root = os.path.join(self.base_image_dir, 'qground_dataset')
        image_path = os.path.join(image_root, 'images', image_info['image'])

        anns = image_info['ann_list']
        # choose from annotations of one person if there are multiple annotations
        anns = random.choice(anns)

        question = None
        gt_answer = None
        text_answers = None

        images = np.array(Image.open(image_path).convert('RGB'))
        ori_size = images.shape[:2]
        # preprocess images for clip
        if self.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(images)
            clip_resize = image_clip.shape[:2]
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
            # clip_image_process(self.clip_image_processor, images)
        else:
            image_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]
        
        images = self.transform.apply_image(images)  # preprocess images for sam
        resize = images.shape[:2]
        masks = []

        # category_ids = [ann['label']['category_id'] for ann in anns]
        # category_ids = list(set(category_ids))
        # sampled_num = min(self.num_classes_per_sample, len(category_ids))
        # sampled_category_ids = np.random.choice(category_ids, size=sampled_num, replace=False)

        sampled_sents = question
        sampled_answers = gt_answer
        sample_text_answers = text_answers

        questions = []
        answers = []
        use_assign_list = []
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token)

        if self.use_quality_description:
            quality_des = np.random.choice([anns['quality_description'], ''])
        else:
            quality_des = ''

        sampled_seg = np.array(Image.open(os.path.join(image_root, 'masks', anns['segmentation_mask'])).convert('RGB'))

        all_seg_masks = get_all_distortion_masks(sampled_seg)
        
        # save masks for debug
        # Image.fromarray(sampled_seg).save('tmp_color_mask.png')
        # for k, v in all_seg_masks.items():
        #     img = Image.fromarray(v.astype(np.uint8) * 255)
        #     img.save(f'tmp_{k}.png')

        if len(all_seg_masks) == 0:
            question_template = random.choice(self.long_question_list)
            question_template = question_template.format(sent=quality_des)
            # question_template = question_template.format(sent='')
            questions.append(question_template)
            answers.append(random.choice(NO_DISTORTION_ANSWER))

        elif len(all_seg_masks) == 1:
            # single answer 
            exist_distortions = list(all_seg_masks.keys())
            category_name = random.choice(exist_distortions)
            question_template = random.choice(self.short_question_list)
            question_template = question_template.replace('\n', f'\n {quality_des}')
            questions.append(question_template.format(class_name=category_name))

            masks.append(get_distortion_mask(sampled_seg, category_name))
            answer_temp = random.choice(self.single_answer_list) if self.seg_token_num == 1 else random.choice(self.single_answer_list).replace('[SEG]', seg_token)
            target_answer = answer_temp.format(class_name=category_name)
            answers.append(target_answer)
            # answers.append(f'{quality_des} {target_answer}')
        else:
            # multi answer 
            question_template = random.choice(self.long_question_list)
            question_template = question_template.format(sent=quality_des)
            # question_template = question_template.format(sent=" ")
            questions.append(question_template)

            target_list = []
            for k, v in all_seg_masks.items():
                masks.append(v)
                target_list.append(k)
            
            separate = random.random() < 0.5
            if separate:
                answer_temp = random.choice(
                    [
                        'The segmentation results are as follows: ',
                        'The segmentation results are: ',
                        'Here are the segmentation results: ',
                        'The results are ',
                    ]
                )
                class_seg_list = []
                for class_name in target_list:
                    class_seg_list.append(f'{class_name} [SEG]')
                target_answer = answer_temp + ', '.join(class_seg_list)
                target_answer = target_answer.replace('[SEG]', seg_token)
            else:
                _seg = ['[SEG]'] * len(target_list)
                if len(target_list) > 1:
                    part1 = ', '.join(_seg[:-1])
                    part2 = ' and ' + _seg[-1]
                    _seg = part1 + part2 
                else:
                    _seg = _seg[0]
            
                answer_temp = random.choice(self.multi_answer_list)
                _answer_temp = answer_temp.format(class_name=', '.join(target_list).lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                _answer_temp = _answer_temp if self.seg_token_num == 1 else _answer_temp.replace('[SEG]', seg_token)
                target_answer = _answer_temp
            
            answers.append(target_answer)
            # answers.append(f'{quality_des} {target_answer}')
        
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1
        
        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous(), self.img_size)

        if len(masks) > 0:
            masks = torch.from_numpy(np.stack(masks, axis=0))
        else:
            mask_shape = images.shape[0]
            masks = torch.zeros(0, mask_shape, mask_shape)

        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        if self.masks_process_with_clip:
            mask_shape = image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)
        
        return (
            image_path,
            images,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            clip_resize,
            questions,
            sampled_sents,
            use_assign_list
        )


def transform_mask(masks, size):
    height, width = masks.shape[-2:]
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    new_shape = (new_long, new_short) if width <= height else (new_short, new_long)
    masks = F.interpolate(masks[None].float(), size=new_shape, mode="nearest")[0].bool()

    orig_height, orig_width = new_shape
    crop_height, crop_width = size, size
    crop_height, crop_width = int(crop_height), int(crop_width)
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width
    assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width
    # if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
    masks = masks[..., top:bottom, left:right]

    return masks


def clip_image_process(clip_image_processor, images):
    # import pdb;pdb.set_trace()
    images = make_list_of_images(images)
    # crop_size = clip_image_processor.crop_size
    images = [convert_to_rgb(image) for image in images]
    images = [to_numpy_array(image) for image in images]
    input_data_format = infer_channel_dimension_format(images[0])
    resize_transform = ResizeLongestSide(clip_image_processor.size['shortest_edge'])
    images = [resize_transform.apply_image(image) for image in images]
    # images = [
    #         to_channel_dimension_format(image, clip_image_processor.data_format) for image in images
    #     ]

    images = [
                clip_image_processor.rescale(image=image, scale=clip_image_processor.rescale_factor)
                for image in images]
    images = [
                    clip_image_processor.normalize(image=image, mean=clip_image_processor.image_mean, std=clip_image_processor.image_std)
                    for image in images
                ]


    images = [
                F.pad(torch.tensor(image).permute(2, 0, 1), (0, 224-image.shape[1], 0, 224-image.shape[0])) for image in images
            ]
    #images[0]: 3, h, w
    return images[0]

if __name__ == "__main__":
    version = '/opt/tiger/PointVIS/LISA/ckpt/Llava-7B-V1-1'
    version = '../PixelLM-7B/hf_model/'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    ret_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids

    dataset = MultiReasonSegDataset('./dataset/', tokenizer, 'openai/clip-vit-large-patch14')
    for i in range(len(dataset)):
        import pdb;pdb.set_trace()
        data = dataset[i]


