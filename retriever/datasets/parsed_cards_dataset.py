import csv
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
from sklearn.model_selection import KFold

class ParsedCardsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length, num_folds=5, fold_idx=0, train=True):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_folds = num_folds
        self.fold_idx = fold_idx
        self.train = train
        
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                html = row["Extractor 1 1"]
                soup = BeautifulSoup(html, 'html.parser')
                # pprint(soup.prettify())
                headers = soup.findAll('h2', class_='a1T8t6')
                
                if len(headers) > 0:
                    header = headers[0]     
                else: 
                    continue 
                question = header.text.replace("\xa0", " ")
                # print(question)
                
                all_answers = soup.findAll('p', class_='acXkgi')
                gathered_answer = []
                for answer in all_answers:
                    gathered_answer.append(answer.text.strip().replace("\xa0", " "))

                self.data.append((question, " ".join(gathered_answer)))
                
        self.data = np.array(self.data, dtype=object)                
        splitter = KFold(n_splits=self.num_folds)
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
    dataset = ParsedCardsDataset("retriever/data/internal_all.csv", tokenizer=tokenizer, max_length=512)
    pprint(dataset[0]["query_input_ids"])