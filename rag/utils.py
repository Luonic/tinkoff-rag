from transformers import AutoTokenizer


TOKENIZER_PATH = 'e5-base-tokenizer'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def count_tokens(text):
    return len(tokenizer.tokenize(text, add_special_tokens=False))


def format_docs_with_score(docs):
    print(len(docs))
    concateneted_docs_string = "\n\n".join('\n'.join([f'Id источника: {i}', f'Релевантность: {score}', doc.metadata['title'], doc.page_content]) for i, (doc, score) in enumerate(docs))
    print(concateneted_docs_string)
    print()
    return concateneted_docs_string
    

def format_docs(docs):
    print(len(docs))
    concateneted_docs_string = "\n\n".join('\n'.join([f'Id источника: {i}', doc.metadata['title'], doc.page_content]) for i, doc in enumerate(docs))
    print(concateneted_docs_string)
    print()
    return concateneted_docs_string
    

def preprocess_text(text):
    text = text.replace('\u2060', '')
    return text