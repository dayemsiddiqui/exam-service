import os
from typing import Optional, Dict
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field

# Load environment variables
load_dotenv()

# Check if API key is in environment
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. Please check your .env file."
    )


class TranslationResult(BaseModel):
    """Structured output for a translation."""

    translation: str = Field(description="The translated word")
    part_of_speech: Optional[str] = Field(
        description="The part of speech of the word in the given context"
    )
    confidence: float = Field(description="Confidence score for the translation (0-1)")
    alternatives: Optional[Dict[str, str]] = Field(
        description="Alternative translations with contexts"
    )


class TranslationService:
    def __init__(self):
        """Initialize the translation service with the Groq language model."""
        # Initialize the Groq model
        self.model = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.1,
        )

        # Define the translation prompt template
        self.translation_prompt = PromptTemplate.from_template("""
        You are a professional language translator. Translate the word: "{word}" from any language to English.
        
        Context: "{context}"
        Word position in the context: {word_index}
        
        Analyze the word in its context and provide the most accurate translation.
        Your response should be structured as a JSON with the following fields:
        - translation: the most accurate English translation
        - part_of_speech: the grammatical function in this context
        - confidence: a number between 0 and 1 representing how confident you are
        - alternatives: a dictionary of alternative translations with example contexts
        
        ONLY respond with the JSON object, no additional text.
        """)

    def translate(self, word: str, context: str = "", word_index: int = 0) -> str:
        """
        Translates a word from any language to English using context if available.

        Args:
            word: The word to translate
            context: The sentence or paragraph containing the word
            word_index: The position of the word in the context

        Returns:
            The English translation
        """
        # If no context is provided, create a simple one
        if not context:
            context = word
            word_index = 0

        # Create the structured chain
        chain = self.translation_prompt | self.model | StrOutputParser()

        try:
            # Execute the chain
            result = chain.invoke(
                {"word": word, "context": context, "word_index": word_index}
            )

            # Parse the result as a TranslationResult
            import json

            parsed_result = json.loads(result)

            # Return just the translation for compatibility with existing code
            return parsed_result.get("translation", word)

        except Exception as e:
            print(f"Translation error: {e}")
            # Fallback to returning the original word
            return word
