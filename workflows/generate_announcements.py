from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable

__all__ = ["Announcer", "Announcement", "generate_listening_exam_announcement"]

# Load environment variables
load_dotenv()


class Announcer(BaseModel):
    """Structured output for a speaker."""

    name: str = Field(description="The name of the announcer")
    gender: str = Field(description="The gender of the announcer")
    opinion: str = Field(
        description="The announcement text in German which should be 5 to 10 sentences long"
    )
    question: str = Field(
        description="A True/False question that can be used to test the listener's understanding of the announcement"
    )
    correct_answer: bool = Field(description="The correct answer to the question")
    explanation: str = Field(
        description="An explanation of the correct answer to the question in English. IMPORTANT: The explanation should be in English, not German."
    )
    english_translation: str = Field(
        description="English translation of the announcement"
    )


class Announcement(BaseModel):
    """Structured output for an announcement."""

    speakers: List[Announcer] = Field(description="The speakers making announcements")


model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1,  # Lower temperature for more consistent announcements
).with_structured_output(Announcement)


@traceable(run_type="llm")
def generate_listening_exam_announcement() -> Announcement:
    background_context = """
    Background Context:
    We are generating a listening exam for the telc B1 German exam.
    """

    exam_context = f"""
    Exam Context:
    Generate a 5 realistic public announcement scenarios. Please make sure the announcements are not too similar to each other and there are variations in situation, context and the people making the announcements.
    Make some of the announcements more complex and longer than others, some should be shorter.
    Prevent examinees from keyword matching e.g "Es gibt heute 10% Rabatt für alle frische Produkte in der Supermarkt, nur am Montag bis 10 Uhr". Then the question would be "Es gibt heute 10% Rabatt für alle frische Produkte in der Supermarkt".
    This would prevent them from just listening for the keywords "10% Rabatt" and "Montag" and "10 Uhr" and "Supermarkt" and answering the question correctly.

    For each announcer, you must provide:
    1. A name for the announcer (or role like "Train Conductor")
    2. Their gender (male/female)
    3. Their announcement in German (5-10 sentences)
    4. A True/False question in German about the announcement
    5. The correct answer to that question (true/false)
    6. An explanation of the correct answer to the question in English
    7. English translation of the announcement

    The announcements should be typical of what you might hear in public settings like:
    - Train stations
    - Airports
    - Public buildings
    - Shopping centers
    - Hospitals
    - Discount Offers in supermarkets
    - Response to phone calls from customers explaining when, where and timings e.g of a movie screening etc
    - Public transportation
    - Schools or universities
    - Museums or cultural institutions

    Example structure:
    {{
        "speakers": [
            {{
                "name": "Bahnhofsansager", 
                "gender": "male",
                "opinion": "Sehr geehrte Fahrgäste, wir möchten Sie darüber informieren...", # Announcement in German
                "question": "Der Zug nach Berlin hat eine Verspätung von 30 Minuten?", # Question in German
                "correct_answer": true, # true or false
                "explanation": "The correct answer is true because the announcement states the train is delayed by 30 minutes", # Explanation in English
                "english_translation": "Dear passengers, we would like to inform you..." # English translation of the announcement
            }},
            {{
                "name": "Emily", 
                "gender": "female",
                "opinion": "Günstige Angebote in der Supermarkt, heute nur 10% Rabatt für alle Produkte", # Announcement in German
                "question": "Es gibt heute 10% Rabatt für alle Produkte in der Supermarkt?", # Question in German
                "correct_answer": true, # true or false
                "explanation": "The correct answer is true because the announcement states there is a 10% discount for all products in the supermarket", # Explanation in English
                "english_translation": "There is a 10% discount for all products in the supermarket today" # English translation of the announcement
            }},
            {{ 
                "name": "Mitarbeiter",
                "gender": "male",
                "opinion": "Wir haben heute eine neue Filme im Kino, wir empfehlen Ihnen die neue Produktion", # Announcement in German
                "question": "Es gibt heute einen neuen Film im Kino", # Question in German
                "correct_answer": true, # true or false
                "explanation": "The correct answer is true because the announcement states there is a new film in the cinema", # Explanation in English
                "english_translation": "There is a new film in the cinema today" # English translation of the announcement
            }}
            {{
                "name": "Wettermann",
                "gender": "male",
                "opinion": "Es ist heute stark bewölkt und es regnet etwas in Berlin aber es wird am Samstag sonnig, und in München ist es sonnig bis Samstag", # Announcement in German
                "question": "Es ist regnerisch in Berlin fur die ganze Woche?", # Question in German
                "correct_answer": false, # true or fal  se
                "explanation": "The correct answer is false because the announcement states it is cloudy and raining in Berlin but it will be sunny during the weekend" # Explanation in English
                "english_translation": "It is cloudy and raining in Berlin for the whole week but it will be sunny during the weekend and in Munich" # English translation of the announcement
            }}
        ]
    }}

    All announcements and questions must be in German.
    Make sure the announcements sound authentic and natural for the context.
    Include specific details like times, gate numbers, or locations when appropriate.
    """

    prompt = PromptTemplate(
        template="""
        {background_context}
        {exam_context}

        Generate a structured announcement following the exact format shown in the example.
        """
    ).format(
        background_context=background_context,
        exam_context=exam_context,
    )

    conversation = model.invoke(prompt)
    return conversation
