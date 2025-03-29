from pydantic import BaseModel


class TranslateResponse(BaseModel):
    word: str
    translation: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"word": "hello", "translation": "hola"},
                {"word": "book", "translation": "reservar"},
            ]
        }
    }
