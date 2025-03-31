from pydantic import BaseModel
from workflows.generate_transcript import Conversation
from typing import Optional


class ListeningExamRequest(BaseModel):
    topic: Optional[str] = None


class ListeningExamResponse(BaseModel):
    conversation: Conversation
