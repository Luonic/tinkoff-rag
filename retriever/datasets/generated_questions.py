import os
import json
from pprint import pprint
from glob import glob

import torch
import numpy as np
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
from sklearn.model_selection import KFold

class GeneratedDataset(Dataset):
    def __init__(self, jsons_dir, tokenizer, max_length, num_folds=5, fold_idx=0, train=True):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_folds = num_folds
        self.fold_idx = fold_idx
        self.train = train
        
        filenames = glob(os.path.join(jsons_dir, "*.json"), recursive=True)
        filenames.sort()
        cards = []
        for filename in filenames:
            with open(filename) as json_f:
                data = json.load(json_f)

            # pprint(data)
            passage = data["source"]
            queries = []
            for query in data["user_queries"]:
                queries.append(query["query"])
                    
            cards.append((queries, passage))


        self.data = np.array(cards, dtype=object)                
        splitter = KFold(n_splits=self.num_folds, shuffle=True, random_state=0)
        for i, (train_index, test_index) in enumerate(splitter.split(self.data)):
            if i == self.fold_idx:
                train_data, val_data = self.data[train_index], self.data[test_index]
                break
        self.data = train_data if self.train else val_data
        self.data = np.array(self.data)
        
        data = []
        for sample in self.data:
            for query in sample[0]:
                data.append((query, sample[1]))
                
        self.data = data

        self.data = self.data[:1000]
        


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
    tokenizer = transformers.AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    dataset = GeneratedDataset(jsons_dir="retriever/data/generated_questions", tokenizer=tokenizer, max_length=512)
    print(tokenizer.convert_ids_to_tokens(dataset[1]["query_input_ids"]))
    print(len(dataset))