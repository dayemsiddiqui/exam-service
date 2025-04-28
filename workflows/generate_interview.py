from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable
import os

__all__ = ["Interview", "generate_interview_transcript"]

# Load environment variables
load_dotenv()


# New model for Interviewer
class Interviewer(BaseModel):
    """Details about the interviewer."""
    name: str = Field(description="The name of the interviewer.")
    gender: str = Field(description="The gender of the interviewer (male/female).")


class Interviewee(BaseModel):
    """Details about the interviewee."""
    name: str = Field(description="The full name of the interviewee.")
    profession: str = Field(description="The profession or role of the interviewee (e.g., Student, Artist, Scientist).")
    gender: str = Field(description="The gender of the interviewee (male/female).")


class ConversationSegment(BaseModel):
    """A segment of the conversation by a single speaker."""
    speaker: str = Field(description="Either 'interviewer' or 'interviewee' to identify who is speaking.")
    text: str = Field(description="The text spoken by this speaker in German.")
    speaker_gender: str = Field(description="The gender of the speaker (male/female).")


class ExamQuestion(BaseModel):
    """A True/False question for the listening exam based on the interview."""
    question_text: str = Field(description="The True/False question in German about the interview content.")
    correct_answer: bool = Field(description="The correct answer (true/false).")
    explanation: str = Field(description="A concise explanation in English justifying the correct answer, referencing the interview.")


class Interview(BaseModel):
    """Structured output for a complete interview including exam questions."""
    interviewer: Interviewer = Field(description="Details of the interviewer.")
    interviewee: Interviewee = Field(description="Details of the person being interviewed.")
    conversation_segments: List[ConversationSegment] = Field(description="List of conversation segments, alternating between interviewer and interviewee.")
    exam_questions: List[ExamQuestion] = Field(description="List of 10 True/False questions based on the interview.")
    english_translation_conversation: str = Field(description="An accurate English translation of the full conversation .")


# Initialize the model with structured output
model = ChatOpenAI(
    model="gpt-4.1-nano-2025-04-14",
    temperature=0.7
).with_structured_output(Interview)


@traceable(run_type="llm")
def generate_interview_transcript() -> Interview:
    """Generates an extensive interview transcript (~100 sentences), with detailed questions and answers for a German B1 listening exam."""

    prompt_template = PromptTemplate.from_template(
        """
IMPORTANT: Your output must be valid JSON with no markdown formatting, code fences, or additional commentary. 

Generate a German B1 level listening exam interview with the following components:
1. An interviewer (with a realistic German name and gender)
2. An interviewee (with a realistic German name, profession, and gender)
3. A LENGTHY conversation in segments (approximately 25-30 turns for each speaker, totaling around 100 sentences)
4. 10 True/False questions based on the interview content
5. English translation of the conversation

IMPORTANT NOTES:
- Both the interviewer and interviewee must have authentic German names, not just "Interviewer" as a placeholder
- Ensure the interviewer and interviewee are not the same person/gender
- The interview should be SIGNIFICANTLY LONGER than a typical dialogue, with at least 50 total segments (25 each for interviewer and interviewee)
- Each segment should contain 1-3 sentences, totaling approximately 100 sentences across the entire conversation
- The interviewer should ask detailed questions about the interviewee's life, career, and personal experiences
- Cover multiple aspects of the interviewee's life to provide rich material for the exam questions
- Use appropriate German B1 level vocabulary and grammar
- Ensure all field names match exactly as shown above
        """
    )

    prompt_value = prompt_template.invoke({})
    return model.invoke(prompt_value)

# Example usage (optional, for testing)
# if __name__ == "__main__":
#     generated_interview = generate_interview_transcript()
#     print("--- Generated Interview ---")
#     print(generated_interview.model_dump_json(indent=2))
#     print("------------------------") 