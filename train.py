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
from torch.distributed.fsdp import MixedPrecision
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

# Imports for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# Set environment variable for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

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
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls

def main():
    # Set environment variables
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    job_id = now()
    args = parse_args()
    print(args)
    cfg = Config(args)

    # Initialize distributed mode (required for FSDP)
    init_distributed_mode(cfg.run_cfg)
    rank = get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    print(task)
    print('here1\n')
    datasets = task.build_datasets(cfg)
    print('here2\n')

    # Build the model
    model = task.build_model(cfg).to(device)
    print('here3\n')

    # Define FSDP wrapping policy for transformer layers (renamed for clarity)
    fsdp_wrap_policy = transformer_auto_wrap_policy(
        module=model,
        recurse=True,           # Recursively wrap transformer layers
        nonwrapped_numel=10000, # Don’t wrap layers with fewer than 10k parameters
        transformer_layer_cls={
            LlamaDecoderLayer,  # LLaMA transformer layer
            CLIPEncoderLayer,   # CLIP-ViT transformer layer (for eva_clip_g)
        }
    )

    # Wrap the model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_wrap_policy,  # Use the explicitly defined policy
        sharding_strategy="FULL_SHARD",     # Shard params, grads, and optimizer states
        device_id=device,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,      # Use FP16 for params (V100 supports FP16)
            reduce_dtype=torch.float16,     # Reduce gradients in FP16
            buffer_dtype=torch.float16      # Buffers in FP16
        ),
        sync_module_states=True             # Ensure all ranks start with same weights
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
