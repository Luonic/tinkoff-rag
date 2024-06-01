import torch
import yaml
import gc
from yaml.loader import SafeLoader
from tqdm import tqdm

from rag_utils import *
from embedding_model import init_embeddings


config = yaml.load(open('config.yaml', 'r'), Loader=SafeLoader)

RETRIEVAL_MODEL_PATHS = config['RETRIEVAL_MODEL_PATHS']
DATA_PATH = config['DATA_PATH']
TOKENIZER_PATH = config['TOKENIZER_PATH']


if __name__ == "__main__":
    llm = init_llm()
    embeddings = [init_embeddings(path, TOKENIZER_PATH, device=torch.device('cuda:0')) for path in tqdm(RETRIEVAL_MODEL_PATHS)]
    gc.collect()
    retriever = init_retriever(DATA_PATH, embeddings)
    gc.collect()
    rag_chain = init_rag_chain(prompt, retriever, llm)
    # rag_chain = init_multiquery_rag_chain(prompt, multiquery_prompt, retriever, llm)
    question = 'Сколько стоит КЭП?'
    result_dict = invoke_rag_chain(rag_chain, question)
    # result_dict = invoke_multiquery_rag_chain(rag_chain, question)
    print(result_dict)