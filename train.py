"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
import functools

# Imports for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_runner_class(cfg):
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls

def clear_gpu_memory():
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def main():
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

    clear_gpu_memory()

    job_id = now()
    args = parse_args()
    print(f"Arguments: {args}")
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    rank = get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    print(f"Task: {task}")
    datasets = task.build_datasets(cfg)

    # Build model with LoRA enabled via config
    model = task.build_model(cfg)

    # Force all parameters and buffers to FP16
    model = model.to(dtype=torch.float16)  # Convert parameters to FP16
    for buffer in model.buffers():
        buffer.data = buffer.data.to(dtype=torch.float16)  # Convert buffers to FP16

    # Freeze all base parameters (LoRA adapters will remain trainable)
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if "lora" not in name.lower():  # Freeze non-LoRA parameters
            param.requires_grad = False
        else:
            trainable_params += param.numel()
        total_params += param.numel()

    # Debug: Print parameter status, counts, and dtypes
    if rank == 0:
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters (LoRA): {trainable_params}")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}, dtype={param.dtype}, device={param.device}")
        for name, buffer in model.named_buffers():
            print(f"Buffer {name}: dtype={buffer.dtype}, device={buffer.device}")


    # Define FSDP wrapping policy with functools.partial
    fsdp_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
            CLIPEncoderLayer,
        },
        recurse=True,
    )
    # Get the device for FSDP
    device = torch.device(f"cuda:{rank}")

    # Wrap with FSDP for sharding across 7 GPUs
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_wrap_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_id=device,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,  # Reinforce FP16 for parameters
            reduce_dtype=torch.float16,  # FP16 for gradient reduction
            buffer_dtype=torch.float16  # FP16 for buffers
        ),
        cpu_offload=CPUOffload(offload_params=True),
        sync_module_states=True,
        use_orig_params=True,  # Allow mixed requires_grad
    )

    if cfg.run_cfg.wandb_log:
        wandb.login()
        wandb.init(project="minigptv", name=cfg.run_cfg.job_name)
        wandb.watch(model)
    

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

if __name__ == "__main__":
    main()
