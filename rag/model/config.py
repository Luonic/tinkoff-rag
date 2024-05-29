import os
from dataclasses import dataclass, field

@dataclass
class Config:
    model_name: str = 'Vikhrmodels/it-5.2-fp16-cp' # 'microsoft/Phi-3-mini-4k-instruct' 'openai-community/gpt2-large' 
    dataset_name: str = 'wikitext'
    dataset_config: str = 'wikitext-2-raw-v1'
    num_texts: int = None
    num_epochs: int = 1
    learning_rate: float = 5e-5
    max_length: int = 2048
    batch_size: int = 8
    log_every: int = 100
    backend: str = 'nccl'
    master_addr: str = '192.168.101.3'
    master_port: str = '12345'
    world_size: int = 2
    rank: int = field(default_factory=lambda: int(os.getenv('RANK', 0)))
    local_rank: int = field(default_factory=lambda: int(os.getenv('LOCAL_RANK', 0)))
    accumulation_step: int = 8
    use_amp: bool = True
    use_hook: bool = True
    use_FA: bool = True
    use_lora: bool = True
    use_checkpointing: bool = True
    lora_dimention: int = 16
    grad_clip=1.0
    use_distributed: bool = True
    profile: bool = True
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
        self.log_every //= self.batch_size
