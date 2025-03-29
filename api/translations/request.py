from pydantic import BaseModel


class TranslateRequest(BaseModel):
    word: str
    context: str
    wordIndex: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "word": "hello",
                    "context": "Can you say hello in Spanish?",
                    "wordIndex": 3,
                },
                {"word": "book", "context": "I need to book a flight.", "wordIndex": 3},
            ]
        }
    }
