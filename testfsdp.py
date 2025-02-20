import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def main():
    # Initialize the distributed process group with NCCL backend for GPU communication
    torch.distributed.init_process_group(backend="nccl")
    
    # Get the rank of the current process and set the corresponding CUDA device
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create a simple model (Linear layer) and move it to the current GPU
    model = torch.nn.Linear(10, 10).to(device)

    # Define the FSDP auto-wrap policy for transformer layers
    policy = transformer_auto_wrap_policy(
        transformer_layer_cls=(LlamaDecoderLayer,),  # Use a tuple for transformer layer classes
        recurse=True,                                # Enable recursive wrapping of submodules
        nonwrapped_numel=10000                       # Minimum parameter threshold for wrapping
    )

    # Debug: Verify that the policy is a callable function
    print(f"Policy type: {type(policy)}, Callable: {callable(policy)}")

    # Wrap the model with FSDP using the corrected policy
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy="FULL_SHARD"  # Fully shard parameters, gradients, and optimizer states
    )

    # Confirm successful wrapping
    print("FSDP wrapping successful")

if __name__ == "__main__":
    main()
