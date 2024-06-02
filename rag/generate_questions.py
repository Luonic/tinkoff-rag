import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
import os, os.path
import re
import uuid
import json
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, pipeline
from concurrent.futures import ThreadPoolExecutor

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

loader = CSVLoader('data/prepared_dataset_for_generation.csv', 
                   source_column='description', 
                   metadata_columns=['url', 'title'], 
                   csv_args={'delimiter': ',', 'quotechar': '"'})

docs = loader.load()

for doc in docs:
    doc.page_content = doc.page_content.replace('description: ', '')

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')

def count_tokens(text):
    return len(tokenizer.tokenize(text, add_special_tokens=False))

def write_json(data):
    path = Path('data/generated_questions') / Path(str(uuid.uuid4()) + '.json')
    with open(path, 'w') as f:
        json.dump(data, f)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=20,
    length_function=count_tokens,
    is_separator_regex=False,
)

splits = text_splitter.split_documents(docs)

prompt = ChatPromptTemplate.from_messages([SystemMessage("""Тебе надо сгенерировать потенциальные вопросы пользователя к документу, который будет представлен ниже.
Вопросов должно быть 10 штук.
Вопросы должны различаться друг от друга, как будто их писал абсолютно разные люди, разного достатка и интеллекта, пола, расы и возраста, запросы должны отличаться длинной, формальностью, стилем написания, грамотностью. Но они должны быть полностью релевантны первоначальному вопросу из базы знаний и ответу, выдумывать какие-то факты запрещено.
Поровну для всех вопросов используй релевантные синонимы к понятиям из вопроса, парафразы, в каких-то вопросах можешь писать с ошибками.

Ответ должен быть только JSON, следующего формата:

{
  "user_queries": [
    {
      "query": "Вопрос пользователя"
    }
    ...
  ]
}

"""), ("user", """Вопрос:\n{question}\n\nОтвет:\n{answer}""")])

llm = ChatOpenAI(model="gpt-4o", openai_proxy=os.getenv('PROXY'))
parser = JsonOutputParser()
chain = prompt | llm | parser

def parse_and_write_split(split):
    split_content = split.page_content
    split_title = split.metadata['title']
    json_answer = chain.invoke({"question": split_title, "answer": split_content})
    json_answer['split_content'] = split_content
    json_answer.update(split.metadata)
    write_json(json_answer)

# for split in tqdm(splits):
with ThreadPoolExecutor(5) as f:
    res = f.map(parse_and_write_split, splits[4:])

for r in tqdm(res):
    print(r)
    