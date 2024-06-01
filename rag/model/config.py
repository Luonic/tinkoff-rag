import os
from dataclasses import dataclass, field

@dataclass
class Config:
    model_name: str = 'Vikhrmodels/it-5.2-fp16-cp' # 'microsoft/Phi-3-mini-4k-instruct' 'openai-community/gpt2-large' 
    save_path: str = 'models_dir'
    log_dir: str = 'tensorboard_logs/metrics_v1'
    path_to_data: str = 'data/train_dataset.json'
    weight_decay: float = 0.01
    test_size: float = 0.1
    test_every: int = 1000 # samples
    num_texts: int = None
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 1000 # samples
    max_length: int = 2048
    batch_size: int = 2
    test_batch_size: int = 2
    log_every: int = 250
    backend: str = 'nccl'
    master_addr: str = '192.168.101.4'
    master_port: str = '12345'
    world_size: int = 2
    rank: int = field(default_factory=lambda: int(os.getenv('RANK', 0)))
    local_rank: int = field(default_factory=lambda: int(os.getenv('LOCAL_RANK', 0)))
    accumulation_step: int = 8
    use_FA: bool = True
    use_hook: bool = True
    use_lora: bool = True
    use_checkpointing: bool = False
    lora_dimention: int = 32
    grad_clip=1.0
    use_distributed: bool = False
    access_token="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"

    def __post_init__(self):
        if not self.use_distributed:
            self.world_size = 1
            self.use_hook = False

        if self.use_FA:
            self.use_amp = False
        if self.use_lora:
            self.use_amp = False
        
        self.learning_rate *= (self.batch_size * self.accumulation_step * self.world_size) ** 0.5
        self.warmup_steps //= self.batch_size * self.accumulation_step * self.world_size
        self.test_every //= self.batch_size * self.world_size
        self.log_every //= self.batch_size * self.world_size


@dataclass
class InferenceConfig():
    model_name: str = 'Vikhrmodels/it-5.2-fp16-cp' # 'microsoft/Phi-3-mini-4k-instruct' 'openai-community/gpt2-large' 
    model_path: str = 'models_dir/model_4'
    max_length: int = 2048
    use_FA: bool = True
    use_lora: bool = True
    lora_dimention: int = 32
    device: str = 'cuda'
    access_token="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"
