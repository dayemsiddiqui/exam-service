from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langsmith import traceable
import os # Import os to access environment variables

__all__ = ["Announcer", "Announcement", "generate_listening_exam_announcement"]

# Load environment variables
load_dotenv()

# Ensure the API key is loaded (optional but good practice for debugging)
# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     print("Warning: GROQ_API_KEY not found in environment variables.")


class Announcer(BaseModel):
    """Structured output for a speaker."""

    name: str = Field(description="The name or role of the announcer (e.g., Bahnhofsansager, Flughafenpersonal)")
    gender: str = Field(description="The gender of the announcer (male/female)")
    opinion: str = Field(
        description="The announcement text in German, typically 3 to 7 sentences long."
    )
    question: str = Field(
        description="A True/False question in German that tests comprehension of a key detail in the announcement."
    )
    correct_answer: bool = Field(description="The correct answer to the question (true/false).")
    explanation: str = Field(
        description="A concise explanation in English of why the answer is correct, referencing the announcement."
    )
    english_translation: str = Field(
        description="An accurate English translation of the German announcement."
    )


class Announcement(BaseModel):
    """Structured output for a list of public announcements for a listening exam."""

    speakers: List[Announcer] = Field(description="A list of 5 distinct announcement scenarios.")


# Initialize the model - Consider llama-3.1-70b-versatile if 8b struggles with consistency
# Using a slightly lower temperature might help consistency if needed, but 1 is fine for variety.
model = ChatGroq(
    # Consider trying 'llama-3.1-70b-versatile' if '8b' struggles with complex instructions
    model="llama-3.1-8b-instant",
    temperature=0.8, # Slightly reduced temperature for better focus
).with_structured_output(Announcement)


@traceable(run_type="llm")
def generate_listening_exam_announcement() -> Announcement:
    """Generates 5 diverse announcement scenarios for a German B1 listening exam."""
    
    # --- Prompt Definition ---
    # Note: Removed the large JSON example block.
    # Instructions clearly state the required fields.
    
    prompt_template = PromptTemplate.from_template(
        """
Generate content for a German B1 level listening exam. Create exactly 5 distinct public announcement scenarios.

**Overall Goal:** Produce realistic audio simulation material for language learners.

**Scenario Requirements:**
1.  **Variety:** Ensure diverse situations (train station, airport, supermarket, museum, school, etc.) and announcer roles/genders. Avoid making all announcements sound the same.
2.  **Complexity:** Include a mix of announcement lengths and complexity. Some should be short and direct, others slightly longer with more details.
3.  **Authenticity:** The German text should sound natural for the given context. Use vocabulary and sentence structures appropriate for B1 level.
4.  **Testing Focus:** Design the True/False question to target a specific detail, inference, or key piece of information, discouraging simple keyword matching. For example, avoid questions that just repeat a full sentence from the announcement.

**Output Structure (per announcement):**
For each of the 5 scenarios, provide the following information adhering strictly to the required structure:
    1.  `name`: The name or role of the announcer (e.g., "Bahnhofsansager", "Museumsführerin", "Supermarkt-Mitarbeiter").
    2.  `gender`: "male" or "female".
    3.  `opinion`: The announcement text in German (approx. 3-7 sentences).
    4.  `question`: A True/False question *in German* testing comprehension of the announcement.
    5.  `correct_answer`: `true` or `false`.
    6.  `explanation`: A concise explanation *in English* justifying the correct answer by referencing the announcement content.
    7.  `english_translation`: An accurate English translation of the German announcement text (`opinion`).

**Example Contexts (Inspiration - do not copy directly):**
-   Train delay announcement with reason and new platform.
-   Airport gate change or boarding information.
-   Supermarket special offer with specific conditions (time limit, product type).
-   Museum tour start time and meeting point change.
-   School announcement about an upcoming event or schedule change.
-   Voicemail message providing requested information (e.g., cinema times, opening hours).

**Constraint Checklist:**
- [ ] 5 distinct announcement scenarios generated.
- [ ] All German text (`opinion`, `question`) is grammatically correct and natural-sounding for B1.
- [ ] All English text (`explanation`, `english_translation`) is accurate.
- [ ] Questions effectively test comprehension beyond simple keyword spotting.
- [ ] Output strictly follows the required structure for structured generation.

Generate the 5 announcements now.
        """
    )

    prompt_value = prompt_template.invoke({}) # No variables needed for this template

    print("--- Sending Prompt to Groq ---")
    # print(prompt_value.to_string()) # Uncomment to see the exact prompt being sent
    print("-----------------------------")

    try:
        conversation = model.invoke(prompt_value)
        # Add basic validation
        if not conversation.speakers or len(conversation.speakers) != 5:
             print(f"Warning: Expected 5 speakers, but got {len(conversation.speakers) if conversation.speakers else 0}")
             # Handle this case if necessary, maybe retry or raise an error
        return conversation
    except Exception as e:
        print(f"Error during model invocation: {e}")
        # You might want to re-raise or handle more gracefully
        raise e
