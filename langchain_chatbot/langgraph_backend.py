from langgraph.graph import StateGraph,START,END, add_messages
from typing import TypedDict,Annotated,List
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")


class ChatState(TypedDict):
    message: Annotated[List[BaseMessage],add_messages]

def chat_node(state : ChatState):
    message = state['message']
    response = llm.invoke(message)

    return {'message' : [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

# add Node
graph.add_node('chat_node',chat_node)


# add edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

# compile graph
chatbot = graph.compile(checkpointer=checkpointer)

# execute graph

