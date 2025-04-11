from fastapi import FastAPI, Body, Query, HTTPException
from fastapi.responses import StreamingResponse
import os
from services.translation_service import TranslationService
from services.listening_exam_service import ListeningExamService
from services.listening_exam_announcement_service import (
    ListeningExamAnnouncementService,
)
from services.interview_service import InterviewService
from api.translations import TranslateRequest, TranslateResponse
from api.listening_exam import (
    ListeningExamResponse,
    AudioGenerationRequest,
    ListeningExamAnnouncementResponse,
    InterviewResponse,
)
from workflows.generate_transcript import Conversation
from workflows.generate_announcements import Announcement
from workflows.generate_interview import Interview
from fastapi.middleware.cors import CORSMiddleware
from services.reading_exam_service import ReadingExamService, ReadingAdvertExamResult
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

# Create listening exam announcement service instance
listening_exam_announcement_service = ListeningExamAnnouncementService()

# Create interview service instance
interview_service = InterviewService()


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
    "/listening-exam/transcript",
    response_model=ListeningExamResponse,
    summary="Generate a listening exam transcript",
    description="Generates a telc B1 level listening exam conversation. If no topic is provided, uses a round-robin selection from predefined topics.",
    response_description="Returns a conversation object containing the generated listening exam content",
)
async def generate_transcript(
    topic: str = Query(
        default=None,
        description="The topic for the listening exam conversation. If not provided, a topic will be automatically selected.",
        example="What are your hobbies?",
        min_length=5,
        max_length=200,
    ),
):
    """
    Generate a listening exam transcript for telc B1.
    Returns a conversation with context, dialogue, questions, and answers.
    If no topic is provided, uses round-robin selection from predefined topics.
    """
    conversation: Conversation = listening_exam_service.generate_transcript(topic=topic)
    return ListeningExamResponse(conversation=conversation)


@app.post(
    "/listening-exam/audio",
    response_class=StreamingResponse,
    summary="Generate audio for a text",
    description="Generates audio for the given text using a voice based on gender and speaker index.",
    response_description="Returns the generated audio file as a streaming response",
)
async def generate_audio(request: AudioGenerationRequest):
    """
    Generate audio for the given text.
    Returns the audio file as a streaming response.
    """
    try:
        # Generate the audio file
        audio_file = listening_exam_service.audio_service.generate_audio(
            text=request.text,
            gender=request.gender,
            speaker_index=request.speaker_index,
        )

        # Open the file in binary mode
        def iterfile():
            with open(audio_file, mode="rb") as file_like:
                yield from file_like

            # Delete the file after sending
            os.unlink(audio_file)

        return StreamingResponse(
            iterfile(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(audio_file)}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/listening-exam/announcement",
    response_model=ListeningExamAnnouncementResponse,
    summary="Generate a listening exam announcement",
    description="Generates a telc B1 level listening exam announcement using a round-robin selection from predefined announcement topics.",
    response_description="Returns a conversation object containing the generated announcement content",
)
async def generate_announcement():
    """
    Generate a listening exam announcement for telc B1.
    Returns a conversation with context, announcement, questions, and answers.
    Uses round-robin selection from predefined announcement types.
    """
    announcement: Announcement = (
        listening_exam_announcement_service.generate_announcement()
    )
    return ListeningExamAnnouncementResponse(announcement=announcement)


# New endpoint for generating interview transcripts
@app.get(
    "/listening-exam/interview",
    response_model=InterviewResponse,
    summary="Generate a listening exam interview",
    description="Generates a telc B1 level listening exam interview focused on the interviewee's life, career, and experiences, along with 10 True/False questions.",
    response_description="Returns an interview object containing the dialogue, interviewee details, and exam questions.",
)
async def generate_interview():
    """
    Generate a listening exam interview for telc B1.
    Returns an interview with dialogue, questions, and answers focused on the interviewee.
    The topic is implicitly derived from the interviewee's profile and experiences.
    """
    try:
        interview_data: Interview = interview_service.generate_interview()
        return InterviewResponse(interview=interview_data)
    except Exception as e:
        # Log the exception for debugging
        print(f"Error generating interview: {e}")
        # Raise an HTTPException to return a proper error response to the client
        raise HTTPException(status_code=500, detail=f"Failed to generate interview: {e}")


@app.get(
    "/reading-exam/advert",
    response_model=ReadingAdvertExamResult,
    summary="Generate a reading exam advert",
    description="Generates a reading exam advert for telc B1. Returns a list of questions and adverts.",
    response_description="Returns a list of questions and adverts.",
)
async def generate_reading_exam_advert():
    """
    Generate a reading exam advert for telc B1.
    """
    reading_exam_service = ReadingExamService()
    exam_result: ReadingAdvertExamResult = await reading_exam_service.get_advert_section()
    return exam_result
