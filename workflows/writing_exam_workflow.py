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

class Letter(BaseModel):
    text: str = Field(description="A fictional German letter or email, which can be formal or informal.")

class Task(BaseModel):
    point: str = Field(description="A point in German that the examinee must address in their response.")

class WritingExam(BaseModel):
    letter: Letter = Field(description="The letter/email stimulus.")
    tasks: List[Task] = Field(description="A list of four points in German that must be addressed in the response.")

# Define the prompt template for the letter writing exam
prompt_template = PromptTemplate(
    template="""
Generate a Telc B1 letter writing exam in German. The output must be valid JSON matching the schema:
{{
  "letter": {{
    "text": "<German letter or email stimulus>"
  }},
  "tasks": [
    {{
      "point": "<First point in German>"
    }},
    {{
      "point": "<Second point in German>"
    }},
    {{
      "point": "<Third point in German>"
    }},
    {{
      "point": "<Fourth point in German>"
    }}
  ]
}}

Important:
- It should be a {letter_type} letter/email.
- The letter/email should be about {letter_topic}.
- The tasks must be exactly four points in German.
- The tasks are points that the examinee must address in their response to the letter/email. They are not questions.
- The tasks represent the questions that the sender of the letter/email would like to know from the examinee.
- Do not include additional keys or metadata.
- Ensure valid JSON matching the Pydantic schema.

### Sample Prompt:
In Telc B1 exam the examinee receive a fictional letter/email (in german) that could be formal or informal that they need to respond to, the exam also contains four points (also in German) that the examinees need to address in their response to the letter/email.
"""
)

class WritingExamWorkflow:
    def __init__(self):
        # Initialize the LLM with structured output based on the Pydantic model
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano-2025-04-14",
            temperature=random.uniform(0.5, 0.7),
            max_retries=2
        ).with_structured_output(WritingExam)

    def letter_type(self) -> str:
        """
        Randomly selects a letter type from a list of predefined types.
        """
        return random.choice(["formal", "informal"])
    
    def letter_topic(self, letter_type: str) -> str:
        formal_topics = [
            "Beschwerde über eine mangelhafte Dienstleistung",
            "Anfrage bezüglich Ihrer Produktpalette",
            "Bewerbung um eine Stelle als Auszubildender",
            "Anfrage nach detaillierten Produktinformationen",
            "Kündigung eines Vertrags gemäß Frist",
            "Bestätigung eines Termins bei Ihrem Unternehmen",
            "Antrag auf Verlängerung einer Zahlungsfrist",
            "Beschwerde über Lärmbelästigung in der Nachbarschaft",
            "Anfrage für mögliche Kooperationsmöglichkeiten",
            "Einladung zu einer geschäftlichen Veranstaltung",
            "Rückmeldung zu Ihrer letzten Rechnung",
            "Ankündigung einer personellen Vertretung während meiner Abwesenheit",
            "Antrag auf Gehaltserhöhung nach erfolgreich abgeschlossenem Projekt",
            "Bestellung von Materialien oder Produkten für Ihr Büro",
            "Anfrage zur Teilnahme an einer Zertifikatsübergabe",
            "Einholen von Angeboten zur Büroausstattung",
            "Anfrage zur Referenzschreibung nach einem Praktikum",
            "Anfrage zur Zusendung der Sitzungsprotokolle",
            "Einsicht in Abrechnungsunterlagen erbitten",
            "Antrag auf Urlaubsvertretung für Kolleginnen und Kollegen",
            "Beschwerde über übermäßige Parkgebühren",
            "Anfrage zu Lieferzeiten Ihrer Produkte",
            "Einladung zur Jahreshauptversammlung",
            "Mitteilung über Änderung Ihrer Bankverbindung"
        ]
        informal_topics = [
            "Einladung zum Geburtstag eines Freundes",
            "Entschuldigung für das Versäumen eines Treffens",
            "Erzählung von deinem letzten Urlaub",
            "Dankeschön für ein schönes Geschenk",
            "Beschreibung deines neuen Haustiers",
            "Nachrichten an einen alten Schulfreund",
            "Einladung zum Grillabend im Garten",
            "Vorstellung deiner neuen Wohnung",
            "Erzählung über deinen Kinobesuch",
            "Vorschlag für einen gemeinsamen Ausflug ins Grüne",
            "Bitte um Hilfe beim Umzug",
            "Bericht von einem schönen Konzertbesuch",
            "Einladung zum Spieleabend bei dir zu Hause",
            "Entschuldigung für eine Verspätung beim Treffen",
            "Grüße von deiner aktuellen Reise",
            "Neuigkeiten von deinem Umzug berichten",
            "Fragen zur Planung deines Junggesellenabschieds",
            "Einladung zum spontanen Kaffeetreffen",
            "Erinnerung an unser Projekt an der Uni",
            "Grüße an deine Familie übermitteln",
            "Tipps für deinen Kochabend austauschen",
            "Neuigkeiten über deinen neuen Job teilen",
            "Entschuldigung für das Verschieben unseres Treffens",
            "Einladung zu einem Spaziergang im Park",
            "Fragen nach Empfehlungen für Filme"
        ]

        if letter_type == "formal":
            return random.choice(formal_topics)
        else:
            return random.choice(informal_topics)

    @traceable(run_type="llm")
    async def generate_writing_exam(self) -> WritingExam:
        """
        Generates a letter writing exam by invoking the LLM with the prompt template.
        """
        type = self.letter_type()   
        topic = self.letter_topic(type)
        prompt = prompt_template.invoke({"letter_type": type, "letter_topic": topic})
        exam = await self.llm.ainvoke(prompt)
        return exam 