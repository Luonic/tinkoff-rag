import torch
import torch.nn.functional as F

from typing import List
from torch import Tensor
from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from langchain_core.embeddings.embeddings import Embeddings


class MyEmbeddings(Embeddings):
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.batch_size = 256

        def average_pool(self,
                         last_hidden_states: Tensor,
                         attention_mask: Tensor) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_dict = self.tokenizer(texts[i: i + self.batch_size], max_length=512, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {key: value.to(self.model.device) for key, value in batch_dict.items()}
                with torch.inference_mode():
                    outputs = self.model(**batch_dict)
                batch_embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.extend(batch_embeddings.tolist())
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return embeddings
        
        def embed_query(self, query: str) -> List[float]:
            return self.embed_documents([query])[0]
            

def init_embeddings(model_path, tokenizer_path, device):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if device.type == 'cuda':
        # config = PeftConfig.from_pretrained(model_path)
        
        model = AutoModel.from_pretrained(
            model_path,
            # config.base_model_name_or_path,
            # load_in_8bit=True,
            torch_dtype=torch.float32,
            device_map=device
        )
        
        # model = PeftModel.from_pretrained(
        #     model,
        #     model_path,
        #     torch_dtype=torch.bfloat16
        # )
    else:
        config = PeftConfig.from_pretrained(model_path)
        
        model = AutoModel.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=True,
            torch_dtype=torch.float32,
            device_map=device
        )
        
        model = PeftModel.from_pretrained(
            model,
            model_path,
            torch_dtype=torch.float32
        )

        
    model.eval()
    hf_embeddings = MyEmbeddings(model, tokenizer)
    return hf_embeddings