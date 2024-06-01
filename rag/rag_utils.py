import os
from typing import List
from operator import itemgetter

from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.load import dumps, loads

from utils import *


# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = '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'
# os.environ['OPENAI_API_KEY'] = 'sk-proj-m5NeFrxHnqtuaBKzp8fgT3BlbkFJLGTGlctha1mHRwKksMIg'


prompt = ChatPromptTemplate.from_messages([
    SystemMessage("Вы помощник для выполнения ответов на вопросы для российского банка Тинькофф, а конкретно одного из его подразделений для бизнеса и предпринимателей - Тинькофф.Бизнес."), 
    ("user", """Используйте следующие части полученного контекста для ответа на вопрос. 
Если в контексте нет ответа на поставленный вопрос, то скажите, что не знаете, либо ведите диалог с пользователем в режиме болталки, если он задает какие-то общие вопросы.
Используйте максимум три предложения и дайте краткий и максимально точный ответ, в соответствии с контекстом.
Вопрос: {question}
Контекст: {context}""")])


multiquery_prompt = ChatPromptTemplate.from_template("""Вы помощник языковой модели искусственного интеллекта для ответов на вопросы из базы знаний, посвященной банковскому обслуживанию юридических лиц и бизнесу в целом. Ваша задача — сгенерировать пять разных версий заданного вопроса пользователя для извлечения соответствующих документов из векторной базы данных. Создавая несколько точек зрения на вопрос пользователя, ваша цель — помочь пользователю преодолеть некоторые ограничения поиска по сходству на основе расстояния. Можно использовать парафразы или синонимы к понятиям из вопроса, но так чтобы они обладали высокой степенью релевантности к первоначальному вопросу. Укажите эти альтернативные вопросы, разделенные символами новой строки. Исходный вопрос: {question}""")



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


def init_retriever(data_path: str, embedding_models: list):
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
    
    retrievers = []
    bm25_retriever = BM25Retriever.from_documents(
        documents=splits,
    )
    bm25_retriever.k = 2
    # retrievers.append(bm25_retriever)
    
    for embedding_model in embedding_models:
        vectorstore = Chroma.from_documents(documents=splits, 
                                            embedding=embedding_model)
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # return chroma_retriever
        retrievers.append(chroma_retriever)
    
    count_retrievers = len(retrievers)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers, weights=[1 / count_retrievers] * count_retrievers
    )
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





#---------Далее в процессе-----------

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]
    

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results
    
def delete_empty_values_from_list(list):
    return [value for value in list if value]
    

def init_multiquery_rag_chain(prompt, multiquery_prompt, retriever, llm):
    
    def extract_question(inputs):
        return inputs['question']
    
    def combine_initial_and_generated_questions(initial_question, generated_questions):
        return [initial_question] + generated_questions
    
    def generate_and_combine(inputs):
        initial_question = extract_question(inputs)
        generated_queries_result = generate_queries({"question": initial_question})
        combined_queries = combine_initial_and_generated_questions(initial_question, generated_queries_result)
        return combined_queries
        
    generate_queries = (
        multiquery_prompt 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    retrieval_chain = (
        {"question": itemgetter("question")} 
        | generate_and_combine
        | delete_empty_values_from_list 
        # | retriever.map() 
        # | reciprocal_rank_fusion 
        # | format_docs_with_score
    )
    return retrieval_chain
    
    # | retriever.map() | reciprocal_rank_fusion | format_docs_with_score
    
    structured_llm = llm.with_structured_output(CitedAnswer)
    
    final_rag_chain = (
        {"context": retrieval_chain, 
         "question": itemgetter("question")} 
        | prompt
        | structured_llm
    )
    
    chain = RunnablePassthrough.assign(
        answer=final_rag_chain
    )
    
    return chain
    # ###
    
    
    # rag_chain_from_docs = (
    # RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    #     | prompt
    #     | structured_llm
    # )
    
    # retrieve_docs = (lambda x: x["question"]) | retriever
    
    # chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    #     answer=rag_chain_from_docs
    # )
    
    # return final_rag_chain
        
def invoke_multiquery_rag_chain(rag_chain, question):
    answer_data = rag_chain.invoke({'question': question})
    return answer_data
    



# def init_rag_chain(prompt, retriever, llm):
#     rag_chain = (
#         {"context": retriever | format_docs, 
#          "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return rag_chain

