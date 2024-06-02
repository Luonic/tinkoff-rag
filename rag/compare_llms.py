import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os, os.path
import re
import uuid
import json
import time
import requests
from argparse import ArgumentParser
from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


metrics_path = Path('rag_metrics/')

if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)

def write_json(data, name):
    path = metrics_path / Path(name + '.json')
    with open(path, 'w') as f:
        json.dump(data, f)


def calculate_execution_time(func, *args, **kwargs):
    """
    Calculate the time taken to execute a function.

    Parameters:
    func (function): The function to execute.
    *args: Arguments to pass to the function.
    **kwargs: Keyword arguments to pass to the function.

    Returns:
    tuple: The result of the function and the time taken to execute it.
    """
    start_time = time.time()  # Record the start time
    result = func(*args, **kwargs)  # Execute the function with the provided arguments
    end_time = time.time()  # Record the end time
    
    execution_time = end_time - start_time  # Calculate the time taken
    return result, execution_time


prompt = ChatPromptTemplate.from_messages([SystemMessage("""Вы - экспертная языковая модель, ваша задача - оценить ответы, сгенерированные Retrieval Augmented Generation (RAG) пайплайном, по следующим критериям:

Логическая правильность: Проверьте, является ли ответ логически согласованным и соответствует ли он здравому смыслу.
Фактологическая точность: Проверьте, нет ли в ответе фактологических ошибок, опираясь на предоставленное таргетное значение.
Общая корректность: Оцените, соответствует ли ответ поставленному вопросу и является ли он релевантным и уместным.
Ясность и четкость: Проверьте, является ли ответ четким и понятным для пользователя.
Полнота: Убедитесь, что ответ охватывает все важные аспекты вопроса и предоставляет полную информацию, если это необходимо.
Инструкции:
Вопрос: Вопрос, на который нужно найти ответ.
Контекст: Информация, на основе которой происходила генерация ответа языковой моделью после retrieval (и он не всегда может совпадать с таргетным значением).
Таргетное значение (golden label): Ответ из первоначального датасета (тот, который лежит в векторной базе), и считается наилучшим ответом на вопрос.
Ответ: Ответ, сгенерированный языковой моделью после RAG пайплайна.
Шаги оценки:
Оцените ответ по каждому из пяти критериев (логическая правильность, фактологическая точность, общая корректность, ясность и четкость, полнота).
Сначала проведите рассуждение по каждому критерию, указав на любые обнаруженные ошибки или несоответствия.
После рассуждения вынесите окончательный вердикт: "да" или "нет".
Дайте развернутую оценку, поясняя свой вердикт.
Итоговый вердикт должен быть представлен в JSON формате: {"output": "да"} или {"output": "нет"}.
Пример:
Вопрос:
Каковы основные требования для открытия расчетного счета для юридического лица?

Контекст:
Для открытия расчетного счета юридическому лицу необходимо предоставить устав компании и паспорт руководителя.

Таргетное значение (golden label):
Основные требования для открытия расчетного счета для юридического лица включают предоставление учредительных документов компании, свидетельства о государственной регистрации, свидетельства о постановке на учет в налоговом органе, а также документов, удостоверяющих личность руководителя компании и лиц, имеющих право подписи.

Ответ:
Для открытия расчетного счета юридическому лицу нужно предоставить только устав компании и паспорт руководителя.

Оценка:
Логическая правильность: Ответ логически неправильный, так как не учитывает все необходимые документы, указанные в таргетном значении.
Фактологическая точность: В ответе содержится фактологическая ошибка: кроме устава компании и паспорта руководителя, необходимы также свидетельства о государственной регистрации, свидетельства о постановке на учет в налоговом органе и документы, удостоверяющие личность лиц, имеющих право подписи.
Общая корректность: Ответ не соответствует таргетному значению и неполный.
Ясность и четкость: Ответ сформулирован ясно, но информация неверна и неполна.
Полнота: Ответ не охватывает все важные аспекты вопроса, так как не упоминает о всех необходимых документах.
Рассуждение:
Ответ не соответствует таргетному значению и содержит фактологическую ошибку, не упоминая о всех необходимых документах для открытия расчетного счета для юридического лица.

Итоговый вывод:
{"output": "нет"}
"""), ("user", """Вопрос:\n{question}\n\nКонтекст\n{context}\n\nТаргетное значение\n{ground_truth}\n\nОтвет:\n{answer}""")])



def get_conclusion(text):
    return text[text.find('{'):]


def invoke_chain(row):
    question = row['question']
    context = '\n'.join(row['context'])
    answer = row['answer']
    ground_truth = row['ground_truth']

    llm_output = chain.invoke({"question": question, "context": context, "answer": answer, "ground_truth": ground_truth})
    return llm_output


def request(query):
    r = requests.post('http://79.120.8.93:31000/assist_with_context', json={'query': query})
    time.sleep(10)
    return r
    

def read_test_data():
    with open('data/test_dataset.json', 'r') as f:
        data = json.load(f)
    return data


def inference_test_data(test_data):
    new_test_data = []
    for row in tqdm(test_data):
        query = row['title']
        result, exec_time = calculate_execution_time(request, query)
        row.update(result.json())
        row['exec_time'] = exec_time
        row['question'] = row.pop('title')
        row['context'] = row.pop('contexts')
        row['answer'] = row.pop('text')
        row['ground_truth'] = row.pop('description')
        llm_output = invoke_chain(row)
        row['llm_output'] = llm_output['output'].lower()
        new_test_data.append(row)
    return new_test_data
    

def calculate_metrics():
    test_data = read_test_data()
    new_test_data = inference_test_data(test_data)
    df = pd.DataFrame(new_test_data)
    llm_output = df['llm_output'].value_counts()
    df['url_in_generated_links'] = df.loc[:, ['url', 'links']].apply(lambda row: row[0] in row[1], axis=1)
    output_dict = {'mean_exec_time': df['exec_time'].mean(), 
                   'llm_metric': dict(df['llm_output'].value_counts(normalize=True)),
                   'url_in_generated_links': dict(df['url_in_generated_links'].value_counts(normalize=True))}
    return output_dict


llm = ChatOpenAI(model="gpt-4o", openai_proxy=os.getenv('PROXY'))
json_parser = JsonOutputParser()
str_parser = StrOutputParser()


chain = prompt | llm | str_parser | get_conclusion | json_parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', help='delimited list input', type=str)
    args = parser.parse_args()
    experiment_name = args.name
    metrics = calculate_metrics()
    write_json(metrics, experiment_name)

