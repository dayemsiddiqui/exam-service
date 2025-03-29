from fastapi import FastAPI
from pydantic import BaseModel
from services.translation_service import TranslationService

app = FastAPI()


class TranslateRequest(BaseModel):
    word: str
    context: str
    wordIndex: int


class TranslateResponse(BaseModel):
    word: str
    translation: str


# Create translation service instance
translation_service = TranslationService()


@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


@app.post("/translate")
async def translate(request: TranslateRequest):
    # Use the translation service to translate the word
    translation = translation_service.translate(request.word)

    return TranslateResponse(word=request.word, translation=translation)
