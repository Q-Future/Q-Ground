import argparse
import os
import shutil
import sys
import time
from functools import partial
import logging
import deepspeed
import numpy as np
import torch
import transformers
import copy
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.PixelLM import PixelLMForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, dict_to_cuda)

import gc

from datetime import datetime

import random


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# from azure.identity import DefaultAzureCredential
def parse_args(args):
    parser = argparse.ArgumentParser(description="PixelLM Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
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
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="pixellm", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--num_classes_per_question", default=1, type=int)
    parser.add_argument("--pad_train_clip_images", action="store_true", default=False)
    parser.add_argument("--masks_process_with_clip", default=False, action="store_true")
    parser.add_argument("--preprocessor_config", default='', type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=224, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--use_expand_question_list", action="store_true", default=False)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)

    parser.add_argument("--save_hf_model_path", type=str, default=None)
    parser.add_argument("--use_multi_scale_mm_projector", action="store_true", default=False)
    parser.add_argument("--mm_vision_select_layer", type=str, default="-11,-2")
    
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--pretrain", action="store_true", default=False)
    
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    args.rank = int(os.getenv('RANK', -1))
    seed_everything(123 * (args.rank + 1))

    if args.rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(os.path.join(args.log_dir, f"tb_{time_tag}"))

        log_filename = os.path.join(args.log_dir, 'meta.log')
        i = 1
        while os.path.exists(log_filename):
            log_filename = os.path.join(args.log_dir, 'meta_{}.log'.format(str(i)))
            i += 1
        logger = logging.getLogger('pixellm_logger')
        logger.setLevel(logging.INFO)


        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(args)
    
    else:
        writer = None
        logger = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if args.seg_token_num*args.image_feature_scale_num == 1:
        num_added_tokens = tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num*args.image_feature_scale_num)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_token_num": args.seg_token_num,
        "logger": logger,
        "tokenizer": tokenizer,
        "local_rank": args.local_rank,
        "pad_train_clip_images": args.pad_train_clip_images,
        "resize_vision_tower": args.resize_vision_tower,
        "resize_vision_tower_size": args.resize_vision_tower_size,
        "vision_tower_for_mask": args.vision_tower_for_mask,
        "separate_mm_projector": args.separate_mm_projector,
        "masks_process_with_clip": args.masks_process_with_clip,
        "image_feature_scale_num": args.image_feature_scale_num,
        "use_multi_scale_mm_projector": args.use_multi_scale_mm_projector,
        "mm_vision_select_layer": [int(i) for i in args.mm_vision_select_layer.split(',')],
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    model = PixelLMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if args.pretrain:
        model.get_model().initialize_pixellm_modules(model.get_model().config)
    
    for p in vision_tower.parameters():
        p.requires_grad = False
    if args.resize_vision_tower_size == 224:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0 and not args.pretrain:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "mask_decoder",
                                "image_feature_neck",
                                "prompt_encoder",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    if args.weight:
        state_dict = torch.load(args.weight, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_list = ["lm_head", "embed_tokens", "prompt_encoder", "image_feature_neck", "mask_decoder", "text_hidden_fcs"]

    if args.resize_vision_tower_size != 224:
        trainable_list.append('mm_projector')
        trainable_list.append('out_mm_projector')
    
    if args.pretrain:
        for p in model.parameters():
            p.requires_grad = False
        trainable_list = ['mm_projector']

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in trainable_list
            ]
        ):
            print("Trainable parameter: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
            p.data = p.data.to(dtype=torch.float32)
    
    if args.pretrain:
        args.vqa_data = 'llava_pretrain'

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
        seg_token_num=args.seg_token_num*args.image_feature_scale_num,
        num_classes_per_question=args.num_classes_per_question,
        pad_train_clip_images=args.pad_train_clip_images,
        masks_process_with_clip=args.masks_process_with_clip,
        preprocessor_config=args.preprocessor_config,
        use_expand_question_list=args.use_expand_question_list,

    )
    print("____seg_token_num in data:________: ", args.seg_token_num*args.image_feature_scale_num)
    print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "communication_data_type": "fp32",

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0,
                "betas": (args.beta1, args.beta2),
                "torch_adam": True
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch * 2, # due to some reason the steps here have to x2 to be correct
                "warmup_num_steps": args.warmup_steps * 2,
                "warmup_type": "linear",
            }
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    resume_ckpt_dir = "latest_ckpt_model"
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "best_ckpt_model")
        if os.path.exists(resume):
            args.resume = resume
    
    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    train_iter = iter(train_loader)

    if args.rank == 0:
        print("Start training")

    for epoch in range(args.start_epoch, args.epochs):
        
        torch.cuda.empty_cache()
        gc.collect()

        # train for one epoch
        model_engine.train()
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )
        model_engine.eval()
                
        save_dir = os.path.join(args.log_dir, resume_ckpt_dir)
        if args.rank == 0:
            print(f'Saving latest checkpoint to {save_dir}')
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)
        
        # save hf model for each epoch 
        if args.save_hf_model_path:
            torch.distributed.barrier()

            if not args.pretrain:
                merge_model = copy.deepcopy(model_engine.module)
                merge_model = merge_model.merge_and_unload()
            else:
                merge_model = model_engine.module

            state_dict = {}
            for k, v in merge_model.state_dict().items():
                if "vision_tower" not in k:
                    state_dict[k] = v.detach().clone()
            
            # save checkpoint to hf format for each epoch 
            hf_ckpt_path = f"{args.save_hf_model_path}_epoch_{epoch}" 
            print(f"{datetime.now()} Saving hf model to {hf_ckpt_path}")
            if args.rank == 0 and os.path.exists(hf_ckpt_path):
                shutil.rmtree(hf_ckpt_path)

            torch.distributed.barrier()
            merge_model.save_pretrained(hf_ckpt_path, state_dict=state_dict)
            tokenizer.save_pretrained(hf_ckpt_path)

            del merge_model
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

        torch.distributed.barrier()
        
        if args.debug and epoch > 5:
            break 

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            data_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        scheduler.step()
        global_step_total = epoch * args.steps_per_epoch + global_step
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
            
            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict, global_step=global_step)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))

            # print("____loss____:", loss, "_____losses.val___", losses.val)
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.rank == 0:
                progress.display(global_step + 1, extra_info=[f'lr: {scheduler.get_last_lr()[0]:.6f}'])
                writer.add_scalar("train/loss", losses.avg, global_step_total)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step_total)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step_total
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step_total
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step_total)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step_total
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step_total
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if args.debug and global_step > 5:
            break

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step_total)

    return train_iter


if __name__ == "__main__":
    main(sys.argv[1:])
