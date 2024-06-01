import streamlit as st
import requests

# App title
st.title("QnA бот Тинькофф Помощь – Бизнес")
# st.set_page_config(page_title="QnA бот Тинькофф Помощь – Бизнес")
starting_message = "Я могу ответить на типовые вопросы про регистрацию бизнеса, кредиты для бизнеса, бизнес-решения, бухгалтерию, продажи и др."

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": starting_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": starting_message}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_response(prompt_input):
    response = requests.post('http://79.120.8.93:31000/assist', json={"query": prompt_input})

    return response

# User-provided prompt
if prompt := st.chat_input("Введите ваш вопрос"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                placeholder = st.empty()
                full_respone = response.json()['text'] + '  \nСсылки:  \n' + '  \n'.join(response.json()['links'])
                placeholder.write(full_respone)
        message = {"role": "assistant", "content": placeholder}
        st.session_state.messages.append(message)