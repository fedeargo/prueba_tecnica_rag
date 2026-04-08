from pydantic import BaseModel, Field
from langchain.tools import tool

class InputQuery(BaseModel):
    query: str = Field(description="La consulta del usuario")

#Definicion de la herramienta
@tool(args_schema=InputQuery)
def RAG_search(query: str):
    """
    Tool para consultar a la base de datos vectorial que contiene la información del pdf
    Args: 
        query: La consulta del usuario
    Returns:
        La respuesta a la consulta
    """
    return "La respuesta a la consulta es: " + query

tools = [RAG_search]