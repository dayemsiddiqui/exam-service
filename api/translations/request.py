from pydantic import BaseModel


class TranslateRequest(BaseModel):
    word: str
    context: str
    wordIndex: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"word": "Haus", "context": "Ich gehe nach Haus.", "wordIndex": 3},
                {
                    "word": "Schule",
                    "context": "Die Kinder gehen zur Schule.",
                    "wordIndex": 4,
                },
            ]
        }
    }
