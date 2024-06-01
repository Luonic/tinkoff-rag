import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, get_peft_model

from model.config import Config
from model.setup import setup_ddp, cleanup_ddp
from data.dataset import load_train_test_data
from model.train import train
from model.eval import evaluate


def main():
    config = Config()
    device = setup_ddp(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.use_FA:
        print("FA model loading")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            token=config.access_token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            token=config.access_token
        )
    if config.use_lora and config.use_checkpointing:
        model.enable_input_require_grads()
    
    if config.use_lora:
        # gpt 2 
        # target_modules=["c_attn", "c_proj", "c_fc"]
        # phi 3 3.8
        # target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        # vikhr 5.2
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=config.lora_dimention,  
            lora_alpha=16,  
            target_modules=target_modules, 
            lora_dropout=0.1, 
            bias="none", 
            task_type="CAUSAL_LM" 
        )
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    if config.use_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)

    train_dataset, test_dataset = load_train_test_data(config.path_to_data, tokenizer, max_length=config.max_length, 
                                                       test_size=config.test_size, num_samples=config.num_texts)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size, rank=config.rank)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, sampler=train_sampler)

    # test_sampler = DistributedSampler(test_dataset, num_replicas=config.world_size, rank=config.rank)
    test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False) # sampler=test_sampler

    for batch in train_dataloader:
        print(batch["input_ids"].size())
        break

    for batch in train_dataloader:
        print(batch["input_ids"].size())
        break

    train(model, train_dataloader, test_dataloader, tokenizer, device, config)
    cleanup_ddp()

if __name__ == "__main__":
    main()

"""
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_addr=192.168.101.4 --master_port=12345 main.py
"""
