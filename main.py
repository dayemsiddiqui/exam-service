from fastapi import FastAPI, Body, Query
from services.translation_service import TranslationService
from services.listening_exam_service import ListeningExamService
from api.translations import TranslateRequest, TranslateResponse
from api.listening_exam import ListeningExamRequest, ListeningExamResponse
from workflows.generate_transcript import Conversation
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

# Create listening exam service instance
listening_exam_service = ListeningExamService()


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


@app.get(
    "/listening-exam-conversation",
    response_model=ListeningExamResponse,
    summary="Generate a listening exam conversation",
    description="Generates a telc B1 level listening exam conversation based on the provided topic. The response includes context, dialogue, questions, and answers.",
    response_description="Returns a conversation object containing the generated listening exam content",
)
async def listening_exam_conversation(
    topic: str = Query(
        default="Is friendship important to you?",
        description="The topic for the listening exam conversation",
        example="What are your hobbies?",
        min_length=5,
        max_length=200,
    ),
):
    """
    Generate a listening exam conversation for telc B1.
    Returns a conversation with context, dialogue, questions, and answers.
    """
    # Generate the conversation using the service
    conversation: Conversation = listening_exam_service.generate_conversation(
        topic=topic
    )

    return ListeningExamResponse(conversation=conversation)
