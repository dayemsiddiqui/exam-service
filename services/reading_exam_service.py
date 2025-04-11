from workflows.generate_reading_adverts import ReadingAdvertExamWorkflow, ReadingAdvertExam, ReadingAdvert
from pydantic import BaseModel
from typing import List
import random

class ReadingAdvertExamResult(BaseModel):
    questions: List[ReadingAdvert]
    adverts: List[str]

class ReadingExamService:
    def __init__(self):
        self.workflow = ReadingAdvertExamWorkflow()

    def get_advert_section(self) -> ReadingAdvertExamResult:
        exam: ReadingAdvertExam = self.workflow.generate_exam()
        questions = exam.questions
        correct_answers = [advert.correct_advert for advert in questions]

        ## Randomly select two wrong answers from the questions
        wrong_answers = [question.wrong_advert for question in questions]
        wrong_answers = random.sample(wrong_answers, 2) 

        ## Combine the correct answer and the two wrong answers
        adverts = correct_answers + wrong_answers

        ## Randomly shuffle the adverts
        random.shuffle(adverts)

        ## Create the exam result
        result = ReadingAdvertExamResult(questions=questions, adverts=adverts)
        return result
