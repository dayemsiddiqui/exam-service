from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable

__all__ = ["Speaker", "Conversation", "generate_listening_exam_transcript"]

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
    explanation: str = Field(
        description="An explanation of the correct answer to the question in English"
    )
    english_translation: str = Field(
        description="English translation of the speaker's opinion"
    )


class Conversation(BaseModel):
    """Structured output for a conversation."""

    speakers: List[Speaker] = Field(
        description="A list containing exactly 5 speakers, no more and no less."
    )


model = ChatOpenAI(
    model="gpt-4.1-nano-2025-04-14",
    temperature=0.3,  # Higher temperature for more creative conversations
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
    Generate a conversation with exactly 5 (IMPORTANT: 5) speakers giving their opinions on this topic.
    The exam/conversation should contain exactly 5 speakers/questions not more not less.
    The TELC B1 exam contains 5 speakers/questions expressing their opinions on the topic.
    Therefore you should also include exactly 5 speakers/questions in the conversation list.

    For each speaker, you must provide:
    1. A name for the speaker
    2. Their gender (male/female)
    3. Their opinion in German (5-10 sentences)
    4. A True/False question in German about their opinion
    5. The correct answer to that question (true/false)
    6. An explanation of the correct answer to the question in English
    7. English translation of the speaker's opinion

    IMPORTANT:
    - Ensure that some answers are true and some are false.
    - Ensure that you generate exactly 5 questions not more not less.
    - THE FINAL LIST MUST CONTAIN EXACTLY 5 SPEAKERS. THIS IS A STRICT REQUIREMENT.

    Example structure:
    {{
        "speakers": [
            {{
                "name": "Anna Schmidt",
                "gender": "female",
                "opinion": "Ich finde das sehr wichtig...", # Opinion in German about the {topic}
                "question": "Anna findet, dass man ...? ",
                "correct_answer": false # true or false
                "explanation": "The correct answer is false because ... # Explanation in English",
                "english_translation": "Anna thinks that ... # English translation of the opinion"
            }},
            // ... more speakers - remember: exactly 5 speakers must be generated in this list ...
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
