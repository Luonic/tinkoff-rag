import os
import gc
from typing import List
from operator import itemgetter

from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.load import dumps, loads

from rag.utils import *



prompt = ChatPromptTemplate.from_messages([
    SystemMessage("Вы помощник для выполнения ответов на вопросы для российского банка Тинькофф, а конкретно одного из его подразделений для бизнеса и предпринимателей - Тинькофф.Бизнес."), 
    ("user", """Используйте следующие части полученного контекста для ответа на поставленный вопрос. Ответ должен быть максимально точным, правильным и релевантным вопросу и контексту. Не совершайте фактологических и логических ошибок относительно контекста и вопроса.
Если в контексте нет ответа на поставленный вопрос, то скажите, что не знаете, либо ведите диалог с пользователем в режиме болталки, если он задает какие-то общие вопросы.
Вопрос: {question}
Контекст: {context}""")])


multiquery_prompt = ChatPromptTemplate.from_messages([SystemMessage("""Вы помощник языковой модели искусственного интеллекта для ответов на вопросы из базы знаний, посвященной банковскому обслуживанию юридических лиц и бизнесу в целом. Ваша задача — сгенерировать пять разных, но в то же время близких по смыслу версий заданного вопроса пользователя для извлечения соответствующих документов из векторной базы данных. Можно использовать парафразы или синонимы к понятиям из вопроса, но так чтобы они обладали высокой степенью релевантности к первоначальному вопросу. Укажите исходный вопрос и 5 альтернативных вопросов в JSON следующего формата: {"queries": ["Исходный вопрос", "Альтернативный вопрос 1", ..., "Альтернативный вопрос 5"]}."""), ("user", "Исходный вопрос: {question}")])



class CitedAnswer(BaseModel):
    """Ответьте на вопрос пользователя, основываясь только на указанных источниках, и укажите использованные источники.
    Если ответа на вопрос в контексте нет, то идентификаторы источников указывать не нужно.
    
    """

    answer: str = Field(
        ...,
        description="Ответ на вопрос пользователя, основанный только на данных источниках.",
    )
    citations: List[int] = Field(
        ...,
        description="""Список целочисленных идентификаторов конкретных источников, которые обосновывают ответ."""
        # Если есть несколько похожих и релевантных документов, то нужно указать эти несколько идентификаторов.""",
    )
    

def init_llm():
    llm = ChatOpenAI(model="gpt-4o", openai_proxy=os.getenv('PROXY'))
    return llm
    

def create_splits(data_path):
    loader = CSVLoader(data_path, 
                   source_column='description', 
                   metadata_columns=['title', 'url'], 
                   csv_args={'delimiter': ',', 'quotechar': '"'})
    docs = loader.load()
    
    for doc in docs:
        doc.page_content = preprocess_text(doc.metadata['source'])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=count_tokens,
    )
    splits = text_splitter.split_documents(docs)
    return splits
    

def init_retriever(data_path: str, embedding_models: list):
    splits = create_splits(data_path)
    
    retrievers = []
    # bm25_retriever = BM25Retriever.from_documents(
    #     documents=splits,
    # )
    # bm25_retriever.k = 2
    # retrievers.append(bm25_retriever)
    
    for embedding_model in embedding_models:
        vectorstore = Chroma.from_documents(documents=splits, 
                                            embedding=embedding_model)
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # return chroma_retriever
        retrievers.append(chroma_retriever)
    
    count_retrievers = len(retrievers)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers, weights=[1 / count_retrievers] * count_retrievers
    )
    gc.collect()
    return ensemble_retriever

    
def init_rag_chain(prompt, retriever, llm):
    structured_llm = llm.with_structured_output(CitedAnswer)
    
    rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | structured_llm
    )
    
    retrieve_docs = (lambda x: x["question"]) | retriever
    
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )

    return chain
    

def invoke_rag_chain(rag_chain, question):
    answer_data = rag_chain.invoke({'question': question})
    print(answer_data)
    text_answer = answer_data['answer'].answer
    citations_id = answer_data['answer'].citations
    links = list(set([context_doc.metadata['url'] for i, context_doc in enumerate(answer_data['context']) if i in citations_id]))
    return {'text': text_answer, 'links': links}

    
def reciprocal_rank_fusion(results: list[list], k=60):
    
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results
    

def init_multiquery_rag_chain(prompt, multiquery_prompt, retriever, llm):
    
    generate_queries = (
        multiquery_prompt 
        | llm
        | JsonOutputParser() 
        | (lambda x: x["queries"])
    )

    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

    structured_llm = llm.with_structured_output(CitedAnswer)
    
    final_rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_with_score(x["context"])))
        | prompt
        | structured_llm
    )

    retrieve_docs = (lambda x: x["question"]) | retrieval_chain

    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=final_rag_chain
    )

    return chain
    
 
def invoke_multiquery_rag_chain(rag_chain, question, return_context):
    answer_data = rag_chain.invoke({'question': question})
    print(answer_data)
    text_answer = answer_data['answer'].answer
    citations_id = answer_data['answer'].citations
    links = list(set([context_doc.metadata['url'] for i, (context_doc, score) in enumerate(answer_data['context']) if i in citations_id]))
    if not return_context:
        return {'text': text_answer, 'links': links}
    else: 
        context = list([' '.join([context_doc.metadata['title'], context_doc.page_content]) for i, (context_doc, score) in enumerate(answer_data['context']) if i in citations_id])
        return {'text': text_answer, 'links': links, 'contexts': context}
