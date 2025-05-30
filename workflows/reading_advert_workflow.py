from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable
import random
from .html_formatter_workflow import HtmlFormatterWorkflow
import asyncio

## Export the workflow
__all__ = ["ReadingAdvertExamWorkflow", "ReadingAdvert", "ReadingAdvertExam"]
## Load environment variables
load_dotenv()



## Define the schema for the output
class ReadingAdvert(BaseModel):
    question: str = Field(description="The question to be answered by the reading advert. In German.")
    correct_advert: str = Field(
    description="The correct advert that matches the question. In German. This should look like a real advert that could be found in a newspaper, magazine, or other media. ")
    wrong_advert: str = Field(description="A wrong advert that sounds similar and plausible to the question but is not the correct answer. In German. This should look like a real advert that could be found in a newspaper, magazine, or other media. ")
    explanation: str = Field(description="An explanation of the correct answer to the question in English. Do not address the wrong answer or any explanation of the wrong answer. IMPORTANT: The explanation should be in English, not German.")

class ReadingAdvertExam(BaseModel):
    questions: List[ReadingAdvert] = Field(description="A list of 10 questions and answers.")

## Define the prompt template
prompt_template = PromptTemplate(
    template="""
    Generate a Telc B2 Leseverstehen Teil 3 exam. This part of the exam has 10 questions and their correct and wrong adverts.
    The questions are based on the adverts.
    The questions are in German.
    The adverts are in German and they might contain information like dates, prices, locations as well as details about the products or services being advertised.
    The adverts should be similar to real life adverts, as they occur in newspapers, magazines, and other media, in terms of language and style. 
    However they shoud be varied and not all adverts should be about the same product or service.
    Ensure that each time this generation is run, the resulting exam is unique and distinct from any previous generations.
    Important Checklist:
    - Each question (aka advert) should be unique and distinct from any previous questions.
    - The adverts should be atleast a few sentences long. 
    - The output should be a JSON, structure output as per the given schema.



    This potential list of topics that could be used for the adverts, but dont have to constrained to these topics:
    {topic_list}

    Here is some free text example on how a real world exam looks likes, so you can some reference for the style and difficulty level. The vocab can be of higher difficulty than the average B2 exam. Though please dont copy the exact questions and answers, but use it as a reference for the style and difficulty level
    There should be significant variance in nature, type, content and style of the adverts:
    {exam_example}
    """
)


class ReadingAdvertExamWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=random.uniform(0.5, 0.7), max_retries=2).with_structured_output(ReadingAdvertExam)

    def get_exam_example(self) -> str:
        with open("examples/advert_exam_example.txt", "r") as file:
            return file.read()
    
    def get_topic_list(self) -> str:
        # This list could be loaded from a file or config if needed
        topic_list = [
            "Auto", "Haus", "Job", "Geld", "Gesundheit", "Reisen", "Mode",
            "Geschichten", "Kunst", "Musik", "Sport", "Technik", "Natur",
            "Geschichte", "Geographie", "Politik", "Gesellschaft", "Medizin",
            "Pädagogik", "Philosophie", "Psychologie", "Religion", "Sprachen",
            "Wirtschaft", "Wissenschaft", "Restaurants", "Bücher", "Filme",
            "Musik", "Sport", "Politik", "Hobbys", "Familie", "Freizeit",
            "Gesundheit",
        ]
        topics = random.sample(topic_list, min(10, len(topic_list))) # Ensure we don't request more samples than available
        return ", ".join(topics)
       
    @traceable(run_type="llm")
    async def generate_exam(self) -> ReadingAdvertExam:
        """Generates a new Reading Advert Exam on each call."""
        # Invoke the prompt template with current data
        prompt_data = prompt_template.invoke({
            "exam_example": self.get_exam_example(), 
            "topic_list": self.get_topic_list()
        })
        
        # Generate the exam directly without caching
        exam = await self.llm.ainvoke(prompt_data)
        
        # Add formatting if needed (commented out for now)
        # formatter = HtmlFormatterWorkflow(additional_instructions="...")
        # Format questions here if needed
        
        return exam

