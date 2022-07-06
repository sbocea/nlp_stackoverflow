from fastapi import FastAPI
from nlp import get_tag
from pydantic import BaseModel

app = FastAPI()


class TextToTag(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/tag")
async def tag(item: TextToTag):
    return get_tag(item.text)

