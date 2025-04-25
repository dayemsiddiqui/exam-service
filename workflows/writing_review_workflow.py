from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable
import random

__all__ = ["WritingExamWorkflow", "WritingExam", "Letter", "Task"]

# Load environment variables for API keys, etc.
load_dotenv()

class UserLetterRequest(BaseModel):
    question: str = Field(description="The letter/email that user got in the exam and needs to respond to.")
    key_points: List[str] = Field(description="The key points that the user needs to address in their response.")
    response: str = Field(description="The letter/email that user has written. It should be a response to the question.")

class Correction(BaseModel):
    original_sentence: str = Field(description="The original sentence from the letter/email that user has written.")
    corrected_sentence: str = Field(description="The corrected sentence from the letter/email that user has written.")
    explanation: str = Field(description="The explanation of the correction that user has written. This should be in English.")

class WrittenExamEvaluation(BaseModel):
    corrections: List[Correction] = Field(description="A list of corrections that user has made to the letter/email. For sentences that are correct, do not include any corrections.")

# Define the prompt template for the letter writing exam
prompt_template = PromptTemplate(
    input_variables=["question", "key_points", "response"],
    template="""
You are a German language teacher for Telc B1 exam. You are given a letter/email that user has written and a list of key points that user needs to address in their response.

Your task is to evaluate the user's response and provide corrections for the user.

This was the letter/email that was given to the user as the question to which they need to respond:
{question}

These are the key points that user needs to address in their response:
{key_points}

This is the letter/email that user has written:
{response}



The output must be valid JSON matching the schema:  
{{
  "corrections": [
    {{
      "original_sentence": "The original sentence from the letter/email that user has written.",
      "corrected_sentence": "The corrected sentence from the letter/email that user has written.",
      "explanation": "The explanation of the correction that user has written. This should be in English."
    }}
  ]
}}

Important:
- Do not include additional keys or metadata.
- Ensure valid JSON matching the Pydantic schema.
- For sentences that are correct, do not include any corrections.

### Context:
In Telc B1 exam the examinee receive a fictional letter/email (in german) that could be formal or informal that they need to respond to, the exam also contains four points (also in German) that the examinees need to address in their response to the letter/email.
"""
)

class WritingReviewWorkflow:
    def __init__(self):
        # Initialize the LLM with structured output based on the Pydantic model
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano-2025-04-14",
            temperature=0.2,
            max_retries=2
        ).with_structured_output(WrittenExamEvaluation)

    @traceable(run_type="llm")
    async def evaluate_written_exam(self, user_letter_request: UserLetterRequest) -> WrittenExamEvaluation:
        """
        Generates a letter writing exam by invoking the LLM with the prompt template.
        """
        # Pass individual fields using simple keys matching the input_variables
        prompt = prompt_template.invoke({
            "question": user_letter_request.question,
            "key_points": user_letter_request.key_points,
            "response": user_letter_request.response,
        })
        evaluation = await self.llm.ainvoke(prompt)
        return evaluation 