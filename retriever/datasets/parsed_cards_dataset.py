import csv
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import Dataset
from bs4 import BeautifulSoup

class ParsedCardsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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

                # print(" ".join(gathered_answer))
                # print()
                # print()
                self.data.append((question, " ".join(gathered_answer)))
                # input()
        
        self.data = np.array(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question, answer = self.data[index]
        print(question)
        print(answer)
        lines = [
            "query: " + question, 
            "passage: " + answer
        ]
        encoded = self.tokenizer(
            lines, padding=True, truncation=True, 
            max_length=self.max_length, return_tensors="pt")
        return encoded
        
        

        
if __name__ == "__main__":
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    dataset = ParsedCardsDataset("retriever/data/internal_all.csv", tokenizer=tokenizer, max_length=512)
    pprint(dataset[0]["input_ids"].shape)