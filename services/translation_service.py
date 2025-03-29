class TranslationService:
    def __init__(self):
        # Simple mock translation for German words
        # In a real application, you would use a translation service or dictionary
        self.translations = {
            "gehen": "to go",
            "hause": "home",
            "wir": "we",
            "nach": "to",
        }

    def translate(self, word: str) -> str:
        """
        Translates a word from German to English

        Args:
            word: The German word to translate

        Returns:
            The English translation or the original word if not found
        """
        return self.translations.get(word.lower(), word)
