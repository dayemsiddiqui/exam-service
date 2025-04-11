from workflows.reading_advert_workflow import ReadingAdvertExamWorkflow, ReadingAdvertExam, ReadingAdvert
from pydantic import BaseModel
from typing import List
import random
import hashlib
class Advert(BaseModel):
    id: str
    text: str   

class AdvertQuestion(BaseModel):
    id: str
    question: str
    correct_advert_id: str
    explanation: str

class ReadingAdvertExamResult(BaseModel):
    questions: List[AdvertQuestion]
    adverts: List[Advert]


class ReadingExamService:
    def __init__(self):
        self.workflow = ReadingAdvertExamWorkflow()

    def to_hash_id(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_advert_section(self) -> ReadingAdvertExamResult:
        exam: ReadingAdvertExam = await self.workflow.generate_exam()
        questions = exam.questions
        correct_answers = [advert.correct_advert for advert in questions]

        ## Randomly select two wrong answers from the questions
        wrong_answers = [question.wrong_advert for question in questions]
        wrong_answers = random.sample(wrong_answers, 2) 

        ## Combine the correct answer and the two wrong answers
        adverts = correct_answers + wrong_answers

        ## Create the adverts
        adverts = [Advert(id=self.to_hash_id(advert), text=advert) for advert in adverts]

        ## Randomly shuffle the adverts
        random.shuffle(adverts)

        ## Create the advert questions
        advert_questions = [AdvertQuestion(id=self.to_hash_id(question.question), question=question.question, correct_advert_id=self.to_hash_id(question.correct_advert), explanation=question.explanation) for question in questions]

        ## Create the exam result
        result = ReadingAdvertExamResult(questions=advert_questions, adverts=adverts)
        return result
