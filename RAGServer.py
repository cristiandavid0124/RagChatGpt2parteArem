# main.py
from fastapi import FastAPI, Query
from RAGProyect import get_rag_response

app = FastAPI(
    title="RAG LangChain Server",
    version="1.0",
    description="Servidor API para RAG usando LangChain y FastAPI",
)
@app.get("/LangChain")
async def rag_endpoint(pregunta: str = Query(..., description="Pregunta para el sistema RAG")):
    response = get_rag_response(pregunta)
    return {"response": response}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)