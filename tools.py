from pydantic import BaseModel, Field
from langchain.tools import tool
from utils.base_vectorial import vector_store
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
import logging

logger = logging.getLogger("agente_RAG.tools")

class InputQuery(BaseModel):
    query: str = Field(description="La consulta del usuario")
    state: Annotated[dict, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]

# Definición de la herramienta
@tool(args_schema=InputQuery)
def RAG_search(query: str, state: dict, tool_call_id: str) -> Command:
    """
    Tool para consultar a la base de datos vectorial que contiene la información del pdf
    Args: 
        query: La consulta del usuario
    Returns:
        La respuesta a la consulta
    """
    try:
        documents = vector_store.as_retriever(query=query, k=3, model="gemini-embedding-001")
        
        # Corrección de la extracción y formato:
        textos_formateados = []
        for doc in documents:
            filename = doc.get('filename', 'Documento desconocido')
            texto = doc.get('text', '')
            textos_formateados.append(f"--- Fuente: {filename} ---\n{texto}\n")

        context = "Utiliza el siguiente contexto para responder a la consulta:\n\n" + "\n".join(textos_formateados)        
        
        return Command(
            update={
                "context": context,
                "messages": [ToolMessage(content=context, tool_call_id=tool_call_id)]
            }
        )
    except Exception as e:
        logger.error(f"Error al realizar la consulta: {str(e)}")
        error_msg = f"Ocurrió un error técnico al buscar en la base de datos: {str(e)}. Pídele disculpas al usuario."
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )

tools = [RAG_search]