import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from model.config import Config
from model.setup import setup_ddp, cleanup_ddp
from model.dataset import load_data
from datasets.dataset import LLMDataset
from model.train import train
from model.profile import train_profiling
from peft import LoraConfig, get_peft_model

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
            lora_alpha=32,  
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

    # dataset = load_data(config, tokenizer)
    dataset = LLMDataset(path_to_data="data/internal_all.csv", tokenizer=tokenizer, 
                         max_length=config.max_length, num_samples=config.num_texts)
    sampler = DistributedSampler(dataset, num_replicas=config.world_size, rank=config.rank)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, sampler=sampler)

    if config.profile:
        train_profiling(model, dataloader, device, config)
    else:
        train(model, dataloader, device, config)
    cleanup_ddp()

if __name__ == "__main__":
    main()

"""
git remote add origin https://Konductor000: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@github.com/Konductor000/https://github.com/Luonic/tinkoff-rag.git
git remote set-url origin git@github.com:Konductor000/https://github.com/Luonic/tinkoff-rag.git
git remote set-url origin https://Konductor000@github.com/Luonic/tinkoff-rag.git
git remote set-url origin https://Konductor000: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@github.com/Luonic/tinkoff-rag.git
"""