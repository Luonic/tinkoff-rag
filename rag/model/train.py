import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook, fp16_compress_hook
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
from time import perf_counter
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, device, config):
    print("train started")
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=len(dataloader))
    scaler = GradScaler()
    if config.use_hook:
        hook = bf16_compress_hook if config.use_FA else fp16_compress_hook
        model.register_comm_hook(state=None, hook=hook)

    if dist.get_rank == 0:
        writer = SummaryWriter(log_dir=config.log_dir)

    start = perf_counter()
    accumulation_timings = []
    reduce_timings = []
    loss_records = []

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            local_start = perf_counter()
            inputs = {key: value.to(device) for key, value in batch.items()}

            if (step + 1) % config.accumulation_step == 0 or (step + 1) == len(dataloader):
                with autocast(enabled=config.use_amp):
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                
                total_loss += loss
                loss_records.append(loss)
                loss /= config.accumulation_step

                if config.use_amp:
                    loss = scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
            else:
                with model.no_sync():
                    with autocast(enabled=config.use_amp):
                        outputs = model(**inputs, labels=inputs['input_ids'])
                        loss = outputs.loss

                    total_loss += loss
                    loss_records.append(loss)
                    loss /= config.accumulation_step

                    if config.use_amp:
                        loss = scaler.scale(loss)

                    loss.backward()

            if dist.get_rank() == 0 and (step + 1) % config.log_every == 0:
                avg_loss = total_loss / config.log_every
                print(f"Epoch: {epoch + 1}, Step: {step + 1}, Avg Loss: {avg_loss:.4f}, Time: {perf_counter()-start}")
                total_loss = 0.0
                writer.add_scalar('Loss/train', avg_loss, epoch * len(dataloader) + step)

            local_end = perf_counter()
            if (step + 1) % config.accumulation_step == 0 or (step + 1) == len(dataloader):
                reduce_timings.append(local_end-local_start)
            else:
                accumulation_timings.append(local_end-local_start)

    end = perf_counter()
    print(f"Full training time for {config.num_epochs * config.num_texts} sample is {end - start}")
    print(f"Mean of reduce timings {sum(reduce_timings) / len(reduce_timings)}")
    if config.accumulation_step > 1:
        print(f"Mean of accumulation timings {sum(accumulation_timings) / len(accumulation_timings)}")
    else:
        print(f"Gradient accumulation diabled")

    if dist.get_rank == 0:
        writer.close()