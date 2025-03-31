from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable

__all__ = ["Speaker", "Conversation", "generate_transcript"]

# Load environment variables
load_dotenv()


class Speaker(BaseModel):
    """Structured output for a speaker."""

    name: str = Field(description="The name of the speaker")
    gender: str = Field(description="The gender of the speaker")
    opinion: str = Field(
        description="The opinion of the speaker, in German which should be 5 to 10 sentences long"
    )
    question: str = Field(
        description="A True/False question that can be used to test the listener's understand of what the speaker said"
    )
    correct_answer: bool = Field(description="The correct answer to the question")


class Conversation(BaseModel):
    """Structured output for a conversation."""

    speakers: List[Speaker] = Field(description="The speakers in the conversation")


model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,  # Higher temperature for more creative conversations
).with_structured_output(Conversation)


@traceable(run_type="llm")
def generate_listening_exam_transcript(topic: str) -> Conversation:
    background_context = """
    Background Context:
    We are generating a listening exam for the telc B1 German exam.
    """

    exam_context = f"""
    Exam Context:
    The topic of the exam is "{topic}".
    Generate a conversation with exactly 5 speakers giving their opinions on this topic.

    For each speaker, you must provide:
    1. A name for the speaker
    2. Their gender (male/female)
    3. Their opinion in German (5-10 sentences)
    4. A True/False question in German about their opinion
    5. The correct answer to that question (true/false)

    Note ensure that some answers are true and some are false.

    Example structure:
    {{
        "speakers": [
            {{
                "name": "Anna Schmidt",
                "gender": "female",
                "opinion": "Ich finde das sehr wichtig...", # Opinion in German about the {topic}
                "question": "Anna findet, dass man ...? # Question in German",
                "correct_answer": false # true or false
            }},
            // ... more speakers ...
        ]
    }}

    All opinions and questions must be in German.
    Each speaker should have a different perspective on the topic.
    """

    prompt = PromptTemplate(
        template="""
        {background_context}
        {exam_context}

        Generate a structured conversation following the exact format shown in the example.
        """
    ).format(
        background_context=background_context,
        exam_context=exam_context,
        topic=topic,
    )

    conversation = model.invoke(prompt)
    return conversation
