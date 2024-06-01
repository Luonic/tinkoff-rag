import transformers.PretrainedTokenizerBase
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # sequence_length = torch.tensor(features["input_ids"]).shape[1]