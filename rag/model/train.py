import torch
import os
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook, fp16_compress_hook
from transformers import get_linear_schedule_with_warmup
from time import perf_counter
from torch.utils.tensorboard import SummaryWriter

try:
    from .eval import evaluate
except:
    from .eval import evaluate


def save_model(model, optimizer, epoch, path):
    # Save the model state, optimizer state, and current epoch to a file
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)


def get_lr(optimizer):
    # Retrieve the current learning rate from the optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, train_dataloader, test_dataloader, tokenizer, device, config):
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, 
                                                num_training_steps=len(train_dataloader) * config.num_epochs / config.accumulation_step / config.world_size)
    # Register communication hook to reduce communication time if specified in the config
    if config.use_hook:
        hook = bf16_compress_hook if config.use_FA else fp16_compress_hook
        model.register_comm_hook(state=None, hook=hook)

    # Initialize TensorBoard writer if the current process is the main process
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=config.log_dir)

    start = perf_counter()
    total_loss = 0.0
    test_steps = 0

    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            test_steps += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Perform gradient accumulation train step
            if (step + 1) % config.accumulation_step == 0 or (step + 1) == len(train_dataloader):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss     
                log_loss = loss.item() 
                total_loss += log_loss          
                loss /= config.accumulation_step

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                with model.no_sync(): # Disable all reduce operation
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    log_loss = loss.item() 
                    total_loss += log_loss      
                    loss /= config.accumulation_step

                    loss.backward()
            
            # Log training metrics to TensorBoard
            if dist.get_rank() == 0:
                current_lr = get_lr(optimizer)
                writer.add_scalar('Loss/train', log_loss, epoch * len(train_dataloader) + step)
                writer.add_scalar('LR', current_lr, epoch * len(train_dataloader) + step)

            if (step + 1) % config.log_every == 0:
                avg_loss = total_loss / config.log_every
                print(f"Epoch: {epoch + 1}, Step: {step + 1}, Avg Loss: {avg_loss:.4f}, Time: {perf_counter()-start}")
                total_loss = 0.0
            
            # Evaluate the model periodically
            if test_steps >= config.test_every or epoch + step == 0:
                test_steps = 0
                with torch.no_grad():
                    metrics = evaluate(model, test_dataloader, tokenizer)
                for metric_name, value in metrics.items():
                    writer.add_scalar(metric_name, value, epoch * len(train_dataloader) + step)

    # Save model
    if dist.get_rank() == 0:
        writer.close()
        os.makedirs(config.save_path, exist_ok=True)
        model_path = f"{config.save_path}/model_{len(os.listdir(config.save_path)) + 1}"
        save_model(model, optimizer, config.num_epochs, model_path)
