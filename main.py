from fastapi import FastAPI, Body
from services.translation_service import TranslationService
from api.translations import TranslateRequest, TranslateResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Translation API",
    description="API for translating words in context",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create translation service instance
translation_service = TranslationService()


@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


@app.post(
    "/translate",
    response_model=TranslateResponse,
    response_description="Returns the translated word",
    summary="Translate a word in context",
    description="Translates a single word from German to English, taking into account the surrounding context.",
)
async def translate(
    request: TranslateRequest = Body(
        ...,
        examples=[
            {
                "word": "Haus",
                "context": "Ich gehe nach Haus.",
                "wordIndex": 3,
            },
            {
                "word": "Schule",
                "context": "Die Kinder gehen zur Schule.",
                "wordIndex": 4,
            },
        ],
    ),
):
    # Use the translation service to translate the word with context
    translation = translation_service.translate(
        word=request.word, context=request.context, word_index=request.wordIndex
    )

    return TranslateResponse(word=request.word, translation=translation)
