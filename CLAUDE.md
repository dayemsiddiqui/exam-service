# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI service that generates German language exam content for Telc B1/B2 exams. The service provides endpoints for generating listening, reading, and writing exam materials with AI-generated content using LangChain and OpenAI.

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running the Server
```bash
# Local development with hot reload
hypercorn main:app --reload

# Production
hypercorn main:app
```

### Environment Variables Required
- `OPENAI_API_KEY` - For LLM-based exam generation
- `ELEVENLABS_API_KEY` - For text-to-speech audio generation
- LangSmith credentials (optional) for tracing

## Architecture

### Three-Layer Structure

1. **API Layer (`api/`)**: Pydantic models for request/response schemas
2. **Service Layer (`services/`)**: Business logic and orchestration of workflows
3. **Workflow Layer (`workflows/`)**: LLM-based generation logic using LangChain

### Key Patterns

**Structured Output**: All workflows use `.with_structured_output()` to enforce Pydantic schemas on LLM responses, ensuring type-safe and predictable outputs.

**Background Caching**: The `utils/caching.py` module provides `AsyncCachedGenerator` for caching expensive LLM-generated content while triggering background updates.

**Round-Robin Topics**: Services like `ListeningExamService` use `itertools.cycle()` to rotate through predefined topics when none is specified.

**Audio Generation**: Two approaches are used:
- ElevenLabs API for conversation/announcement audio (via `AudioService`)
- OpenAI TTS for interview audio with streaming support (via `AudioListeningInterviewExamService`)

### Exam Types

**Listening Exams**:
- **Transcript**: 5 speakers giving opinions on a topic with True/False questions
- **Announcement**: Public announcement (train station, supermarket, etc.) with questions
- **Interview**: Extended interview (~100 sentences) with 10 True/False questions

**Reading Exams**:
- **Advert**: Match questions to classified ads
- **Match Titles**: Match titles to text paragraphs (B2 level)
- **Comprehension**: 5-paragraph text with 5 multiple-choice questions

**Writing Exams**:
- **Letter Writing**: Generate formal/informal letter stimulus with 4 task points
- **Letter Review**: Evaluate user's written response and provide corrections

**Translation**:
- Word translation with context (German ↔ English)
- Sentence translation (German ↔ English)

### LLM Configuration

Most workflows use `gpt-4.1-nano-2025-04-14` model with:
- Temperature: 0.2-0.7 depending on creativity needs
- `@traceable(run_type="llm")` decorator for LangSmith integration
- Structured output using Pydantic models

### Service Instantiation Pattern

Most service instances are created at endpoint level (inside route handlers) rather than module level to avoid state sharing issues. Example:
```python
@app.get("/writing-exam")
async def generate_writing_exam():
    writing_exam_service = WritingExamService()
    exam = await writing_exam_service.get_writing_exam()
    return exam
```

### Audio Handling

Audio files are:
- Stored temporarily in `audio/` directory
- Streamed to clients via `StreamingResponse`
- Deleted after streaming completes using `os.unlink()`
- Use gender-based voice selection with multiple voice IDs per gender

### Error Handling

Services implement fallback mechanisms:
- Return structured error responses matching expected schemas
- Log errors to console with descriptive messages
- Raise `HTTPException` with appropriate status codes and details
