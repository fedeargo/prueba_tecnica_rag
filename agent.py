from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

from tools import tools
from utils.memory import FirestoreSaver

from pydantic import Field
import uuid

#Modelo de Chat VertexAI
llm_gemini = ChatVertexAI(
    model_name = 'gemini-2.5-flash',
    location = 'us-east1',
    temperature=0.0
)

#Modelo con las herramientas embedidas
llm_with_RAG = llm_gemini.bind_tools(tools)

#Estado del agente
class AgentState(MessagesState):
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4())) #Para identificar el hilo de la conversacion

prompt = """	
Eres un asistente de IA que responde preguntas sobre las tendencias que puedan inspirar la creación de nuevos productos o servicios dentro de la organización, generando valor para el negocio y una ventaja competitiva significativa frente a los competidores locales.
"""

#Función para el nodo del agente
def agent_node(state: AgentState):
    system_message = SystemMessage(content=prompt)
    messages = [system_message] + state["messages"]
    response = llm_with_RAG.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

#Nodo de herramientas
tools_node = ToolNode(tools)

# Edge condicional de herramientas 
def tools_condition(state: AgentState):
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get("messages", [])):
        ai_message = messages[-1]
    elif messages := getattr(state, "messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

#Definicion del grafo del agente
#Builder
builder_agente_RAG = StateGraph(AgentState)

#Nodos
builder_agente_RAG.add_node("agent", agent_node) #Nodo del agente
builder_agente_RAG.add_node("tools", tools_node) #Nodo de herramientas

#Edges
builder_agente_RAG.add_edge(START, "agent")
builder_agente_RAG.add_edge("tools", "agent")

#Edge condicional
builder_agente_RAG.add_conditional_edges(
    "agent",
    tools_condition, 
    {"tools": "tools","__end__": END}
    )

#Checkpoint
checkpoint_saver = FirestoreSaver(database="agente-rag-db", collection_name="checkpoints", pw_collection_name="checkpoint_writes")

#Compilacion del grafo
graph_agente_RAG = builder_agente_RAG.compile(checkpointer=checkpoint_saver)

#Función para ejecutar el grafo
def execute_graph(thread_id: str, message: str):
    input_message = HumanMessage(content=message)

    configurable = {
        "metadata" : {"thread_id": thread_id,"doc_id": "GallagherRe Global Insurtech Report 2024-Q4 (1)"},
        "configurable": {"thread_id": thread_id}
    }

    response = graph_agente_RAG.invoke(input={"messages": [input_message]}, configurable=configurable)

    output = response.get("response")

    # Si no hay response (por alguna razón), usar el último mensaje como fallback
    if output is None:
        ultimo_mensaje = response.get("messages", [])[-1]
        response = {
            "message": ultimo_mensaje.content if hasattr(ultimo_mensaje, 'content') else str(ultimo_mensaje)
        }
    
    return output

