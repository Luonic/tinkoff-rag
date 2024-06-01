import json
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
from sklearn.model_selection import KFold

class JSONDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length, num_folds=5, fold_idx=0, train=True):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_folds = num_folds
        self.fold_idx = fold_idx
        self.train = train
        
        with open(json_path, 'r') as file:
            data = json.load(file)['data']
            for example in data:
                question = example['title']
                response = example['description']

                self.data.append((question, response))
                
        self.data = np.array(self.data, dtype=object)                
        splitter = KFold(n_splits=self.num_folds, shuffle=True, random_state=0)
        for i, (train_index, test_index) in enumerate(splitter.split(self.data)):
            if i == self.fold_idx:
                train_data, val_data = self.data[train_index], self.data[test_index]
                break
        self.data = train_data if self.train else val_data
        self.data = np.array(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question, answer = self.data[index]
        # print(question)
        # print(answer)
        lines = [
            "query: " + question, 
            "passage: " + answer
        ]
        encoded = self.tokenizer(
            lines, padding="max_length", truncation=True, 
            max_length=self.max_length, return_tensors="pt")
        query_input_ids = encoded["input_ids"][0]
        query_attention_mask = encoded["attention_mask"][0]
        passage_input_ids = encoded["input_ids"][1]
        passage_attention_mask = encoded["attention_mask"][1]
        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "passage_input_ids": passage_input_ids,
            "passage_attention_mask": passage_attention_mask
        }
        
        
if __name__ == "__main__":
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    dataset = ParsedCardsDataset("data/dataset.json", tokenizer=tokenizer, max_length=512)
    pprint(dataset[0]["query_input_ids"])