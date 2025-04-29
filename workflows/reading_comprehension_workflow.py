from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable
import random

## Export the workflow
__all__ = ["ReadingComprehensionWorkflow", "ReadingComprehensionQuestion", "ReadingComprehensionExam"]
## Load environment variables
load_dotenv()

## Define the schema for the output
# Removed ReadingComprehensionOption as it's no longer needed

class ReadingComprehensionQuestion(BaseModel):
    paragraph_index: int = Field(description="The 0-based index of the paragraph this question refers to (0 for the first paragraph, 1 for the second, etc.).")
    question_text: str = Field(description="The multiple-choice question based on the corresponding paragraph, in German.")
    correct_answer: str = Field(description="The text of the single correct answer in German.")
    wrong_answers: List[str] = Field(description="A list containing exactly two incorrect but plausible answer texts in German.")
    explanation: str = Field(description="An explanation in English why the `correct_answer` is right, referencing the text. Do not explain the wrong answers.")

class ReadingComprehensionExam(BaseModel):
    topic: str = Field(description="The topic of the generated text.")
    full_text: str = Field(description="A long-form text in German, consisting of exactly 5 paragraphs.")
    questions: List[ReadingComprehensionQuestion] = Field(description="A list of 5 multiple-choice questions, one for each paragraph of the text (indices 0-4).")


## Define the prompt template
prompt_template = PromptTemplate(
    template="""
    Generate a Telc B1 Leseverstehen Teil 1 exam component. This involves creating a single continuous text divided into 5 distinct paragraphs, followed by 5 multiple-choice questions, one for each paragraph.

    Text Requirements:
    - The text should be in German and discuss the topic: {topic}.
    - The text must consist of exactly 5 paragraphs.
    - The total length of the text must be between 1500 and 2000 words in total. 
    - Each paragraph should be substantial, with at least 300 words, to ensure a detailed, in-depth article akin to a long newspaper feature or blog post.
    - Please ensure you meet these word counts strictly; too short or too brief content will not be accepted.
    - The language level can incorporate B2 or C1 vocabulary to increase difficulty.
    - The overall text should be cohesive and well-structured, resembling a real newspaper article or blog post in style.

    Question Requirements:
    - Generate exactly one multiple-choice question for each of the 5 paragraphs. Associate each question with its paragraph using a 0-based index (0 for the first paragraph, 4 for the last).
    - For each question, provide:
        - The question text in German (`question_text`).
        - The single correct answer text in German (`correct_answer`).
        - A list of exactly two incorrect but highly plausible answer texts in German (`wrong_answers`). These wrong answers should be closely related to the paragraph's content to mislead someone who hasn't fully understood the nuances.
    - Questions should test deep comprehension and attention to detail, not just superficial keyword matching. It should be challenging to determine the correct answer without careful reading.
    - Provide a brief explanation in English (`explanation`) for why the `correct_answer` is right, referencing the relevant part of the text.

    Output Format:
    - Structure the output as a JSON object according to the provided schema.
    - Ensure the `full_text` contains the complete 5-paragraph text meeting the word count requirements.
    - Ensure the `questions` list has exactly 5 entries, corresponding to paragraphs 0 through 4, each following the `ReadingComprehensionQuestion` schema with `correct_answer` and two `wrong_answers`.

    Topic for this exam: {topic}
    """
)


class ReadingComprehensionWorkflow:
    def __init__(self):
        # Using gpt-4o-mini for cost-effectiveness and speed, adjust if needed.
        self.llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=random.uniform(0.5, 0.7), max_retries=2).with_structured_output(ReadingComprehensionExam)

    def get_topic(self) -> str:
        # Reusing a simplified topic list mechanism, adaptable as needed.
        topic_list = [
            "Umweltschutz im Alltag", "Moderne Arbeitswelt", "Gesunde Ernährung",
            "Reisen und Tourismus", "Technologie und Gesellschaft", "Kulturelle Veranstaltungen",
            "Freizeitaktivitäten", "Bildungssysteme", "Soziale Medien", "Stadtleben vs. Landleben",
            "Ehrenamtliche Tätigkeiten", "Migration und Integration", "Klimawandel", "Nachhaltigkeit",
            "Familienleben", "Freundschaft", "Hobbys", "Sportarten", "Musikrichtungen", "Filmgeschichte"
        ]
        return random.choice(topic_list)

    @traceable(run_type="llm")
    async def generate_exam(self) -> ReadingComprehensionExam:
        """Generates a new Reading Comprehension Exam section based on a random topic."""
        selected_topic = self.get_topic()

        prompt_data = prompt_template.invoke({"topic": selected_topic})

        exam_data = await self.llm.ainvoke(prompt_data)

        # Basic validation (can be expanded)
        if len(exam_data.questions) != 5:
            print(f"Warning: Expected 5 questions, but got {len(exam_data.questions)} for topic '{selected_topic}'.")
            # Potentially add retry logic here if needed
        for i, q in enumerate(exam_data.questions):
            if len(q.wrong_answers) != 2:
                 print(f"Warning: Question {i} (Paragraph {q.paragraph_index}) for topic '{selected_topic}' should have exactly 2 wrong answers, but found {len(q.wrong_answers)}.")
            if q.paragraph_index != i:
                 print(f"Warning: Question {i} has unexpected paragraph index {q.paragraph_index}.")


        return exam_data

# Example usage (optional, for testing)
# if __name__ == "__main__":
#     import asyncio
#
#     async def main():
#         workflow = ReadingComprehensionWorkflow()
#         exam = await workflow.generate_exam()
#         print(f"Generated Exam on Topic: {exam.topic}")
#         print("\nFull Text:")
#         print(exam.full_text)
#         print("\nQuestions:")
#         for i, q in enumerate(exam.questions):
#             print(f"\n--- Question {i+1} (Paragraph Index {q.paragraph_index}) ---")
#             print(f"Q: {q.question_text}")
#             # Combine correct and wrong answers for display, maybe shuffle them
#             options = [q.correct_answer] + q.wrong_answers
#             random.shuffle(options) # Shuffle to simulate MCQ format
#             for idx, opt_text in enumerate(options):
#                 print(f"   {chr(ord('a') + idx)}) {opt_text}")
#             print(f"Correct Answer Text: {q.correct_answer}") # For verification
#             print(f"Explanation: {q.explanation}")
#
#     asyncio.run(main()) 