from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable


## Export the workflow
__all__ = ["ReadingAdvertExamWorkflow", "ReadingAdvert", "ReadingAdvertExam"]
## Load environment variables
load_dotenv()



## Define the schema for the output
class ReadingAdvert(BaseModel):
    question: str = Field(description="The question to be answered by the reading advert. In German.")
    answer: str = Field(description="The correct advert that matches the question. In German.")
    wrong_answer: str = Field(description="A wrong advert that sounds similar and plausible to the question but is not the correct answer. In German.")
    explanation: str = Field(description="An explanation of the correct answer to the question in English. Do not address the wrong answer or any explanation of the wrong answer.")

class ReadingAdvertExam(BaseModel):
    questions: List[ReadingAdvert] = Field(description="A list of 10 questions and answers.")

## Define the prompt template
prompt_template = PromptTemplate(
    template="""
    Generate a Telc B1 Leseverstehen Teil 3 exam. This part of the exam has 10 questions and their correct and wrong answers.
    The questions are based on the adverts.
    The questions are in German.
    The adverts are in German and they might contain information like dates, prices, locations as well as details about the products or services being advertised.
    The adverts should be similar to real life adverts, as they occur in newspapers, magazines, and other media, in terms of language and style. 
    However they shoud be varied and not all adverts should be about the same product or service.
    """
)


class ReadingAdvertExamWorkflow:
    def __init__(self):
        self.llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7).with_structured_output(ReadingAdvertExam)
        self.chain = prompt_template | self.llm 

    @traceable(run_type="llm")
    def generate_exam(self) -> ReadingAdvertExam:
        exam = self.chain.invoke({})
        return exam

