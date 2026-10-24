import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_core import llm, prepare_answer

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

history = []  

SOURCES_MARKER = "\x00SOURCES\x00" 


class Ask(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/chat")
def chat(payload: Ask):
    question = payload.question.strip()
    messages, sources = prepare_answer(question, history)

    def generate():
        answer = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                answer += chunk.content
                yield chunk.content
        history.append((question, answer))
        yield SOURCES_MARKER + json.dumps([{"name": n, "page": p} for n, p in sources])

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/clear")
def clear():
    history.clear()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
