import os
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
import uvicorn
from agent import execute_graph


from agent import graph_agente_RAG

app = FastAPI()
puerto = os.getenv("PORT", 8080)    

#Configuración de CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Modelo de entrada
class Input(BaseModel):
    thread_id: str = Field(example="thread_test_123" ,description="El hilo de la conversacion")
    message: str = Field(exampl="¡Hola!, ¿Quién eres?",description="El mensaje del usuario")

#Modelo de respuesta
class Response(BaseModel):
    response: str = Field(description="La respuesta del agente")

@app.post("/chat")
async def chat(input: Input):
    thread_id = input.thread_id
    message = input.message
    try:
        response = execute_graph(thread_id, message)
        return Response(response=response)
    except Exception as e:
        return Response(response=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(puerto))