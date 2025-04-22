import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Check if API key is in environment
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. Please check your .env file or set it directly."
    )


class SentenceTranslationRequest(BaseModel):
    text: str = Field(description="The sentence to translate.")

class SentenceTranslationResponse(BaseModel):
    translation: str = Field(description="The translated sentence.")


class SentenceTranslationService:
    def __init__(self):
        """Initialize the sentence translation service with the Groq language model."""
        # Initialize the Groq model - using a model suitable for generation/translation
        self.model = ChatOpenAI(
            model="gpt-4.1-nano-2025-04-14", # Using a larger model for potentially better translation
            temperature=0.2, # Slightly higher temp for translation creativity
        )

        # Define prompt templates
        self.en_to_de_prompt = PromptTemplate.from_template("""
        Translate the following English sentence accurately into German.
        English sentence: "{sentence}"
        German translation:
        """)

        self.de_to_en_prompt = PromptTemplate.from_template("""
        Translate the following German sentence accurately into English.
        German sentence: "{sentence}"
        English translation:
        """)

    async def translate_en_to_de(self, text: str) -> str:
        """
        Translates an English sentence to German.

        Args:
            text: The English sentence to translate.

        Returns:
            The German translation.
        """
        try:
            prompt = self.en_to_de_prompt.format(sentence=text)
            result = await self.model.ainvoke(prompt)
            # Extract the content from the AIMessage object
            return result.content.strip()
        except Exception as e:
            print(f"English to German translation error: {e}")
            # Fallback or raise specific error
            return f"Error translating: {text}"

    async def translate_de_to_en(self, text: str) -> str:
        """
        Translates a German sentence to English.

        Args:
            text: The German sentence to translate.

        Returns:
            The English translation.
        """
        try:
            prompt = self.de_to_en_prompt.format(sentence=text)
            result = await self.model.ainvoke(prompt)
            # Extract the content from the AIMessage object
            return result.content.strip()
        except Exception as e:
            print(f"German to English translation error: {e}")
            # Fallback or raise specific error
            return f"Error translating: {text}" 