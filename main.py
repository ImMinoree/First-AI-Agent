from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


load_dotenv()

llm = init_chat_model(
    "gemini-3.8-flash",
    model_provider="google_genai"
)

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classifynif the message requires an emotional (therapist) or logical response."
    )

# set state for graph

class  State(TypedDict):
    messages: Annotated[list, add_messages]  #messages is type list - add new message to the list(user message and llm giving assistance message)
    message_type: str | None
    next: str| None


#define a graph builder
graph_builder = StateGraph(State)

#nodes
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def classify_message(state:State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
             "role": "system",
             "content": """Classify the user message as either:
             - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
             - 'logical': if it asks for facts, information, logical analysis, or practical solutions 
             """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}

def router(state:State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return{"next": "therapist"}

    return {"next": "logical"}


def therapist_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content":"""
                You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked.
                """
        },
        {
            "role": "user", "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content":"""
                You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.
                """
        },
        {
            "role": "user", "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


graph_builder.add_node("chatbot", chatbot)

#need a start node and end note

# graph_builder.add_edge(START, "chatbot") #endkey
graph_builder.add_edge("chatbot", END) #statrtkey

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier") #statrtkey
graph_builder.add_edge("classifier", "router") #statrtkey
graph_builder.add_conditional_edges(
    "router",#source
    lambda state: state.get("next"),
    {"therapist":"therapist", "logical":"logical"}#path_map
)


graph_builder.add_edge("therapist", END )
graph_builder.add_edge("logical", END )

graph = graph_builder.compile() #run


def run_chatbot():
    state = {"messages":[], "message_type":None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]

            content = last_message.content

            if isinstance(content, list):
                print(f"Assistant: {content[0]['text']}")
            else:
                print(f"Assistant: {content}")

if __name__ == "__main__":
    run_chatbot()

# user_input = input("Enter a message: ")
# # state = graph.invoke({"messages": [{"role":"user", "content": user_input}]})

# # response = state["messages"][-1]
# # print(response.content[0]["text"])

# response = llm.invoke([
#     {
#         "role": "system",
#         "content": (
#             "Respond in clean plain text for a terminal. "
#             "Do not use Markdown or LaTeX formatting. "
#             "Keep the answer clear and concise."
#         )
#     },
#     {
#         "role": "user",
#         "content": user_input
#     }
# ])

# print(response.content[0]["text"])