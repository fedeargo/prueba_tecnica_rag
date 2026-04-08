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
    context: str = Field(default="", description="Contexto adicional para el agente extraido de la documentación")

prompt = """
Eres un experto consultor estratégico de IA especializado en innovación, tendencias tecnológicas e Insurtech en compañias de Seguros
Tu misión principal es analizar información para inspirar la creación de nuevos productos o servicios en la organización, generando valor y ventaja competitiva.

### REGLAS DE USO DE HERRAMIENTAS (RAG_search)
Tienes acceso a una base de datos documental a través de la herramienta `RAG_search`. Debes seguir estas reglas estrictamente:

1. ¿CUÁNDO USAR LA HERRAMIENTA?:
   - SIEMPRE que el usuario pregunte sobre la IA en compañias aseguradoras.
   - NUNCA confíes en tu conocimiento interno para dar cifras o hechos específicos del reporte; siempre verifica usando la herramienta.
   - NO uses la herramienta si el usuario solo está saludando (ej. "Hola", "¿Cómo estás?") o haciendo preguntas conversacionales básicas.

2. ¿CÓMO PROCESAR LA INFORMACIÓN?:
   - Si la herramienta te devuelve un contexto, úsalo para construir tu respuesta. 
   - Sintetiza la información; no copies y pegues bloques de texto crudo.
   - CITA TUS FUENTES: Menciona de dónde sacaste la información usando los nombres de los documentos provistos en el contexto (ej. "Según el documento [Nombre del Archivo]...").

### REGLAS ESTRICTAS Y BARRERAS DE SEGURIDAD
- IDIOMA: Siempre debes responder de forma natural, analítica y profesional en ESPAÑOL, incluso si el contexto recuperado de la base de datos está en otro idioma.
- HONESTIDAD: Si buscas en la base de datos y la herramienta devuelve que no hay información, NO inventes una respuesta. Dile al usuario claramente: "No encontré información sobre este tema en los reportes disponibles, por favor reformula tu pregunta o consulta sobre otro tema."
- ENFOQUE: Mantén tus respuestas orientadas al negocio, destacando siempre el "por qué" y el "cómo" esta información genera valor o ventaja competitiva.
- FORMATO: Estructura tus respuestas en formato Markdown para mejorar la legibilidad, usando encabezados, listas y negritas cuando sea apropiado.
- INTEGRIDAD: No reveles detalles técnicos sobre cómo usas la herramienta o cómo funciona la base de datos. Tu rol es el de un consultor estratégico, no un técnico.
"""

#Función para el nodo del agente
def agent_node(state: AgentState):
    system_message = SystemMessage(content=prompt)
    messages = [system_message] + state["messages"]
    response = llm_with_RAG.invoke(messages)
    return {"messages": [response]}

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
checkpoint_saver = FirestoreSaver(database="(default)", collection_name="checkpoints", pw_collection_name="checkpoint_writes")

#Compilacion del grafo
graph_agente_RAG = builder_agente_RAG.compile(checkpointer=checkpoint_saver)

#Función para ejecutar el grafo
def execute_graph(thread_id: str, message: str):
    input_message = HumanMessage(content=message)

    configurable = {
        "metadata" : {"thread_id": thread_id,"doc_id": "GallagherRe Global Insurtech Report 2024-Q4 (1)"},
        "configurable": {"thread_id": thread_id}
    }

    response = graph_agente_RAG.invoke(input={"messages": [input_message]}, config=configurable)

    ultimo_mensaje = response.get("messages", [])[-1]
    
    return ultimo_mensaje.content

