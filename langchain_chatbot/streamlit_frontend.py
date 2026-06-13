from email import contentmanager

from langchain_core.messages import HumanMessage
import streamlit as st
from langgraph_backend import chatbot

CONFIG = {'configurable': {'thread_id': 1}}


# session_state -> dict -> message is not erase if we press enter. It is erased when we reload the page
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# {'role': 'user','content':'hii'}
# {'role':'assistant','content':'hello'}


user_input = st.chat_input("Type Here")

if user_input:

    # first add the message to message history
    st.session_state['message_history'].append({'role': 'user','content':user_input})

    with st.chat_message('user'):
        st.text(user_input)

    response = chatbot.invoke({'message':[HumanMessage(content=user_input)]},config=CONFIG)
    ai_message = response['message'][-1].content
    # first add the message to message history
    st.session_state['message_history'].append({'role': 'assistant','content':ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)

