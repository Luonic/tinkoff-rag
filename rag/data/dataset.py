from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import json
from sklearn.model_selection import train_test_split

def chat_template(question, answer):
    sample = f"""<s> [INST] {question} [/INST]\n{answer} </s>"""

    return sample


class LLMDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, num_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = dataset
        if num_samples:
            self.dataset = self.dataset[:num_samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        header = item['prompt']
        content = item['prediction']

        sample = chat_template(header, content)
        # print(sample);print("-"*100)

        chat_tokens = self.tokenizer(
            sample,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'text': sample,
            'input_ids': chat_tokens['input_ids'].squeeze(),
            'attention_mask': chat_tokens['attention_mask'].squeeze()
        }
    
def load_train_test_data(path_to_data, tokenizer, max_length, test_size=0.1, num_samples=None):
    with open(path_to_data, 'r') as f:
        dataset = json.load(f)

    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=42)
    train_llm_dataset = LLMDataset(train_dataset, tokenizer, max_length, num_samples=num_samples)
    test_llm_dataset = LLMDataset(test_dataset, tokenizer, max_length, num_samples=None)

    return train_llm_dataset, test_llm_dataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = LLMDataset(path_to_data="data/train_dataset.json", tokenizer=tokenizer, max_length=2048)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']



    


