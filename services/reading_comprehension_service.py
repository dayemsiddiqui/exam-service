from workflows.reading_comprehension_workflow import ReadingComprehensionWorkflow, ReadingComprehensionExam, ReadingComprehensionQuestion
from pydantic import BaseModel, Field
from typing import List
import random
import hashlib
import asyncio

class ComprehensionOption(BaseModel):
    id: str
    text: str

class ComprehensionQuestion(BaseModel):
    id: str
    paragraph_index: int # 0-based index
    question_text: str
    options: List[ComprehensionOption] # Shuffled options (a, b, c)
    correct_option_id: str
    explanation: str

class ReadingComprehensionResult(BaseModel):
    topic: str
    full_text: str
    paragraphs: List[str] = Field(description="The full_text split into individual paragraphs.")
    questions: List[ComprehensionQuestion]

class ReadingComprehensionService:
    def __init__(self):
        self.workflow = ReadingComprehensionWorkflow()

    def to_hash_id(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_comprehension_section(self) -> ReadingComprehensionResult:
        """Generates and formats the reading comprehension section."""
        exam_data: ReadingComprehensionExam = await self.workflow.generate_exam()

        # Split full_text into paragraphs by blank lines
        paragraphs = [p.strip() for p in exam_data.full_text.split("\n\n") if p.strip()]

        formatted_questions: List[ComprehensionQuestion] = []

        for q_data in exam_data.questions:
            # Combine correct and wrong answers
            all_answer_texts = [q_data.correct_answer] + q_data.wrong_answers

            # Create ComprehensionOption objects with hash IDs
            options = [
                ComprehensionOption(id=self.to_hash_id(text), text=text)
                for text in all_answer_texts
            ]

            # Shuffle the options for presentation
            random.shuffle(options)

            # Get the hash ID of the correct answer
            correct_option_id = self.to_hash_id(q_data.correct_answer)

            # Create the formatted question
            formatted_q = ComprehensionQuestion(
                id=self.to_hash_id(q_data.question_text), # Use question text for question ID
                paragraph_index=q_data.paragraph_index,
                question_text=q_data.question_text,
                options=options,
                correct_option_id=correct_option_id,
                explanation=q_data.explanation
            )
            formatted_questions.append(formatted_q)

        # Sort questions by paragraph index
        formatted_questions.sort(key=lambda q: q.paragraph_index)

        return ReadingComprehensionResult(
            topic=exam_data.topic,
            full_text=exam_data.full_text,
            paragraphs=paragraphs,
            questions=formatted_questions
        ) 