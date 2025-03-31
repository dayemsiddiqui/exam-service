from pydantic import BaseModel
from workflows.generate_transcript import Conversation
from typing import Optional


class ListeningExamRequest(BaseModel):
    topic: Optional[str] = None


class ListeningExamResponse(BaseModel):
    conversation: Conversation


class AudioGenerationRequest(BaseModel):
    text: str
    gender: str
    speaker_index: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Ich finde Freundschaft sehr wichtig.",
                    "gender": "female",
                    "speaker_index": 0,
                }
            ]
        }
    }
