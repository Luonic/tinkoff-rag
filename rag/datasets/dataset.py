from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from transformers import AutoTokenizer


def parse_html_by_header(html_string, link):
    soup = BeautifulSoup(html_string, 'html.parser')
    current_section = {'card_link': link, 'content': []}
    seen_texts = set()

    for element in soup.descendants:
        if not element.string:
            continue
        text = element.string.strip().replace('\xa0', ' ')
        if text in seen_texts:
            continue

        if element.name == 'h2':
            current_section['header'] = text
            seen_texts.add(text)
        elif element.name == 'a':
            current_section['content'].append({
                'text': text,
                'link': element.get('href')
            })
            seen_texts.add(text)
        elif element.name is None:
            current_section['content'].append({
                'text': text,
                'link': None
            })
            seen_texts.add(text)

    return current_section


def chat_template(question, answer, system=None):
    if system == None:
        system = """
Вы являетесь помощником Тинькофф банка. Вам предоставляется извлеченная информация из документа и запрос пользователя. Ваша задача - предоставить точный и полезный ответ, основываясь на информации из документа. Если информация в документе недостаточна для ответа, сообщите об этом пользователю.
        """
    sample = f"""
<s>[INST] <<SYS>>
{system}
<</SYS>> 
{question} [/INST] 
{answer} </s>
    """
    print(sample)

    return sample


class LLMDataset(Dataset):
    def __init__(self, path_to_data, tokenizer, max_length, num_samples=None):
        self.dataset = pd.read_csv(path_to_data)
        self.dataset = self.dataset.dropna(subset=['Extractor 1 1'])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = [parse_html_by_header(self.dataset["Extractor 1 1"][i], self.dataset["Address"][i]) for i in range(len(self.dataset))]
        if num_samples:
            self.samples = self.samples[:num_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        header = item['header']
        content = item['content']

        content_texts = [c['text'] for c in content]
        content_paragraph = "\n".join(content_texts)
        sample = chat_template(header, content_paragraph)

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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = LLMDataset(path_to_data="data/internal_all.csv", tokenizer=tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']



    


