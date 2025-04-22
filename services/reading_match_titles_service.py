from workflows.reading_match_titles_workflow import ReadingMatchTitleWorkflow, ReadingMatchTitle
from pydantic import BaseModel
from typing import List
import random
import hashlib

class TitleOption(BaseModel):
    id: str
    title: str

class MatchTitleQuestion(BaseModel):
    text: str
    correct_title: str
    explanation: str

class ReadingMatchTitleResult(BaseModel):
    questions: List[MatchTitleQuestion]
    titles: List[TitleOption]

class ReadingMatchTitlesService:
    def __init__(self):
        self.workflow = ReadingMatchTitleWorkflow()

    def to_hash_id(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_match_title(self) -> ReadingMatchTitleResult:
        question: ReadingMatchTitle = await self.workflow.generate_match_title()
        text = question.text
        correct_title = question.correct_title
        wrong_title = question.wrong_title
        explanation = question.explanation

        # Prepare title options and shuffle
        titles_list = [correct_title, wrong_title]
        titles = [TitleOption(id=self.to_hash_id(t), title=t) for t in titles_list]
        random.shuffle(titles)

        # Prepare the question object
        question = MatchTitleQuestion(
            text=text,
            correct_title=correct_title,
            explanation=explanation,
        )

        return ReadingMatchTitleResult(questions=[question], titles=titles)