from langchain_groq import ChatGroq
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
    Each question (aka advert) should be unique and distinct from any previous questions.
    Sometimes you can generate similar questions, to make it more challenging for the examinee to identify the correct answer for each question. But not all questions should be similar.
    The output should be a JSON, structure output as per the given schema.



    This potential list of topics that could be used for the adverts, but dont have to constrained to these topics:
    {topic_list}

    Here is some free text example on how a real world exam looks likes, so you can some reference for the style and difficulty level. The vocab can be of higher difficulty than the average B2 exam. Though please dont copy the exact questions and answers, but use it as a reference for the style and difficulty level
    There should be significant variance in nature, type, content and style of the adverts:
    {exam_example}
    """
)


class ReadingAdvertExamWorkflow:
    def __init__(self):
        self.llm = ChatGroq(model="qwen-qwq-32b", temperature=random.uniform(0.5, 0.7), max_retries=2).with_structured_output(ReadingAdvertExam)
        self.chain = prompt_template | self.llm 

    def get_exam_example(self) -> str:
        with open("examples/advert_exam_example.txt", "r") as file:
            return file.read()
    
    def get_topic_list(self) -> str:
        self.topic_list = [
            "Auto",
            "Haus",
            "Job",
            "Geld",
            "Gesundheit",
            "Reisen",
            "Mode",
            "Geschichten",
            "Kunst",
            "Musik",
            "Sport",
            "Technik",
            "Natur",
            "Geschichte",
            "Geographie",
            "Politik",
            "Gesellschaft",
            "Medizin",
            "Pädagogik",
            "Philosophie",
            "Psychologie",
            "Religion",
            "Sprachen",
            "Wirtschaft",
            "Wissenschaft",
            "Restaurants",
            "Bücher",
            "Filme",
            "Musik",
            "Sport",
            "Politik",
            "Hobbys",
            "Familie",
            "Freizeit",
            "Gesundheit",
        ]
        ## Get a random 10 topics from the list
        topics = random.sample(self.topic_list, 10)
        return ", ".join(topics)
       

    @traceable(run_type="llm")
    async def generate_exam(self) -> ReadingAdvertExam:
        # Note: self.chain.invoke is synchronous, assuming it's okay for the initial exam generation.
        # If self.chain also supports ainvoke, that could be awaited too.
        exam = self.chain.invoke({"exam_example": self.get_exam_example(), "topic_list": self.get_topic_list()})
        formatter = HtmlFormatterWorkflow(additional_description="""
            You are formatting an advert for a reading exam. Make sure to format and style the advert such that it looks like a real advert that could be found in a newspaper, magazine, or other media.
            Things like dates, prices, locations as well as details should be formatted and styled to look like a real advert.
        """)

        async def format_question_adverts(question):
            # Format both adverts concurrently for a single question
            task_correct = formatter.format_html(question.correct_advert)
            task_wrong = formatter.format_html(question.wrong_advert)
            
            formatted_correct, formatted_wrong = await asyncio.gather(
                task_correct, 
                task_wrong
            )
            
            question.correct_advert = formatted_correct.formatted_text
            question.wrong_advert = formatted_wrong.formatted_text
            # No need to return question as it's modified in-place

        ## Create formatting tasks
        tasks = []
        for question in exam.questions:
            tasks.append(format_question_adverts(question))
        
        # Run all question formatting tasks concurrently
        await asyncio.gather(*tasks)
                
        return exam

