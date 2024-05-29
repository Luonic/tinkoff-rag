from torch.utils.data import Dataset
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        return {key: value.squeeze(0) for key, value in inputs.items()}

def load_data(config, tokenizer):
    dataset = load_dataset(config.dataset_name, config.dataset_config, split='train')
    texts = [sample for sample in dataset['text'] if len(sample) > 0][:config.num_texts]
    print(f"Loaded {len(texts)} texts")
    return TextDataset(texts, tokenizer, config.max_length)