import os
import torch
import torch.distributed as dist

def setup_ddp(config):
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
    os.environ['NCCL_SOCKET_IFNAME'] = 'ens192'
    dist.init_process_group(backend=config.backend, rank=config.rank, world_size=config.world_size)
    torch.cuda.set_device(config.local_rank)
    device = torch.device("cuda", config.local_rank)
    print(f"Using device: {device}")
    return device

def cleanup_ddp():
    dist.destroy_process_group()
    