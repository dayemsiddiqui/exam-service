from workflows.reading_match_titles_workflow import ReadingMatchTitleWorkflow, ReadingMatchTitle
from pydantic import BaseModel
from typing import List
import random
import hashlib
import asyncio

class TitleOption(BaseModel):
    id: str
    title: str

class MatchTitleQuestion(BaseModel):
    text: str
    correct_title_id: str
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
        # Generate 5 questions in parallel
        generated_questions: List[ReadingMatchTitle] = await asyncio.gather(
            *[self.workflow.generate_match_title() for _ in range(5)]
        )

        questions_list: List[MatchTitleQuestion] = []
        all_titles_list: List[str] = []

        for question in generated_questions:
            # Collect question details
            questions_list.append(
                MatchTitleQuestion(
                    text=question.text,
                    correct_title_id=self.to_hash_id(question.correct_title),
                    explanation=question.explanation,
                )
            )
            # Collect titles
            all_titles_list.append(question.correct_title)
            all_titles_list.append(question.wrong_title)

        # Prepare title options with hashed IDs and shuffle
        unique_titles = list(set(all_titles_list))
        titles = [TitleOption(id=self.to_hash_id(t), title=t) for t in unique_titles]
        random.shuffle(titles)

        return ReadingMatchTitleResult(questions=questions_list, titles=titles)