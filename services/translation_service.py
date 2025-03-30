import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field

# Load environment variables
load_dotenv()

# Check if API key is in environment
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. Please check your .env file or set it directly."
    )


class TranslationResult(BaseModel):
    """Structured output for a translation."""

    translation: str = Field(description="The translated word in English")
    part_of_speech: Optional[str] = Field(
        description="The part of speech of the word in the given context"
    )
    confidence: float = Field(description="Confidence score for the translation (0-1)")
    alternatives: Optional[List[str]] = Field(
        description="Alternative translations with contexts"
    )


class TranslationService:
    def __init__(self):
        """Initialize the translation service with the Groq language model."""
        # Initialize the Groq model
        self.model = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
        )

        # Create a structured output model
        self.structured_model = self.model.with_structured_output(TranslationResult)

        # Define the translation prompt template
        self.translation_prompt = PromptTemplate.from_template("""
        You are a professional language translator. Translate the word: "{word}" from German to English.
        
        Context: "{context}"
        Word position in the context: {word_index}
        
        Analyze the word in its context and provide the most accurate translation.
        """)

    def translate(self, word: str, context: str = "", word_index: int = 0) -> str:
        """
        Translates a word from any language to English using context if available.

        Args:
            word: The German word to be translated into English
            context: The sentence or paragraph containing the word
            word_index: The position of the word in the context

        Returns:
            The English translation
        """
        # If no context is provided, create a simple one
        if not context:
            context = word
            word_index = 0

        try:
            # First format the prompt
            prompt = self.translation_prompt.format(
                word=word, context=context, word_index=word_index
            )

            # Then invoke the structured model with the formatted prompt
            result = self.structured_model.invoke(prompt)

            # Return just the translation for compatibility with existing code
            print(result)
            return result.translation

        except Exception as e:
            print(f"Translation error: {e}")
            # Fallback to returning the original word
            return word
