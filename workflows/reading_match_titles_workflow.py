from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
import random

## Export the workflow
__all__ = ["ReadingMatchTitleWorkflow", "ReadingMatchTitle"]
## Load environment variables
load_dotenv()



## Define the schema for the output
class ReadingMatchTitle(BaseModel):
    text: str = Field(description="A german text paragraph about any topic. The text should be atleast 15 sentences long.")
    correct_title: str = Field(
    description="The correct title that matches the text. In German.")
    wrong_title: str = Field(description="A wrong title that sounds extremely plausible title but is not the correct answer.")
    explanation: str = Field(description="An explanation of the correct answer to the question in English. Do not address the wrong answer or any explanation of the wrong answer. IMPORTANT: The explanation should be in English, not German.")


## Define the prompt template
prompt_template = PromptTemplate(
    template="""
    Generate a Telc B2 Leseverstehen Teil 1 exam. Feel free to use advance B2 or C1 level vocabulary. 
    The text should be of real world difficulty, like that occurring in newspapers, magazines, articles, blogs, reports or other media.
    Important Checklist:
    - The output should be a JSON, structure output as per the given schema.



    This potential list of topics that could be used for the text, but dont have to constrained to these topics:
    {topic_list}

    """
)


class ReadingMatchTitleWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=random.uniform(0.5, 0.7), max_retries=2).with_structured_output(ReadingMatchTitle)

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
    async def generate_match_title(self) -> ReadingMatchTitle:
        """Generates a new Reading Match Title on each call."""
        # Invoke the prompt template with current data
        prompt_data = prompt_template.invoke({
            "topic_list": self.get_topic_list()
        })
        
        # Generate the exam directly without caching
        exam = await self.llm.ainvoke(prompt_data)
        
        print(exam)
        
        return exam

