from fastapi import APIRouter, Depends, HTTPException
from services.sentence_translation_service import (
    SentenceTranslationService,
    SentenceTranslationRequest,
    SentenceTranslationResponse,
)

router = APIRouter()


# Dependency injection for the service
def get_sentence_translation_service():
    return SentenceTranslationService()


@router.post(
    "/translate/en-to-de",
    response_model=SentenceTranslationResponse,
    summary="Translate English to German",
    tags=["Translation"],
)
async def translate_english_to_german(
    request: SentenceTranslationRequest,
    service: SentenceTranslationService = Depends(get_sentence_translation_service),
):
    """Translates an English sentence to German."""
    try:
        translation = await service.translate_en_to_de(request.text)
        return SentenceTranslationResponse(translation=translation)
    except Exception as e:
        # Log the exception details here if needed
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/translate/de-to-en",
    response_model=SentenceTranslationResponse,
    summary="Translate German to English",
    tags=["Translation"],
)
async def translate_german_to_english(
    request: SentenceTranslationRequest,
    service: SentenceTranslationService = Depends(get_sentence_translation_service),
):
    """Translates a German sentence to English."""
    try:
        translation = await service.translate_de_to_en(request.text)
        return SentenceTranslationResponse(translation=translation)
    except Exception as e:
        # Log the exception details here if needed
        raise HTTPException(status_code=500, detail=str(e)) 