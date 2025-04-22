from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
import random

## Export the workflow
__all__ = ["ReadingMatchTitleWorkflow", "ReadingMatchTitle"]
## Load environment variables
load_dotenv()



## Define the schema for the output
class ReadingMatchTitle(BaseModel):
    text: str = Field(description="A german text paragraph about any topic. The text should be atleast 15 sentences long.")
    correct_title: str = Field(
    description="The correct title that matches the text. In German.")
    wrong_title: str = Field(description="A wrong title that sounds extremely plausible title but is not the correct answer.")
    explanation: str = Field(description="An explanation of the correct answer to the question in English. Do not address the wrong answer or any explanation of the wrong answer. IMPORTANT: The explanation should be in English, not German.")


## Define the prompt template
prompt_template = PromptTemplate(
    template="""
    Generate a Telc B2 Leseverstehen Teil 1 exam. Feel free to use advance B2 or C1 level vocabulary. The text should be of real world difficulty.
    Important Checklist:
    - The output should be a JSON, structure output as per the given schema.
    - The wrong title should be extremely plausible, and should be very similar to the correct title.
    
    
    SUPER IMPORTANT:
    - It should be hard/difficult to guess the correct title, just by keyword matching.
    - The title should not contain obvious keywords from the text, which would make it too easy to guess.
    - The examinee must pay attention to the small details of the text, in order to figure out the correct title.



    The topic of the text should be:
    {topic_list}

    """
)


class ReadingMatchTitleWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=random.uniform(0.5, 0.7), max_retries=2).with_structured_output(ReadingMatchTitle)

    def get_topic_list(self) -> str:
        # Diese Liste könnte bei Bedarf aus einer Datei oder Konfiguration geladen werden
        topic_list = [
            # Gesellschaft & Kultur
            "Stadtentwicklung", "Social-Media-Trends", "Kulturerbe-Erhaltung", 
            "Demografischer Wandel", "Menschenrechtsfragen", "Ethische Dilemmata der modernen Gesellschaft",
            "Gleichstellungsbewegungen", "Einwanderungspolitik", "Krisen im öffentlichen Gesundheitswesen", 
            "Ehrenamt und gemeinnützige Arbeit", "Minderheitensprachen", "Volksbräuche",
            "Interkulturelle Kommunikation", "Generationenkonflikte", "Urbane Lebensstile",
            "Soziale Ungleichheit", "Datenschutz im digitalen Zeitalter", "Fake News und Medienkompetenz",

            # Wissenschaft & Technologie
            "Ethik der künstlichen Intelligenz", "Erneuerbare Energiequellen", "Raumfahrtmissionen",
            "Fortschritte in der Gentechnik", "Cybersicherheitsbedrohungen", "Quantencomputing-Konzepte",
            "Anwendungen der Biotechnologie", "Entwicklungen in der Nanotechnologie", "Nachhaltige Landwirtschaft",
            "Auswirkungen des Klimawandels", "Ozeanografische Entdeckungen", "Durchbrüche in der Teilchenphysik",
            "Robotik im Alltag", "3D-Druck Technologien", "Big Data Analyse", "Smart Home Systeme",
            "Medizintechnische Innovationen", "Batterietechnologien",

            # Kunst & Geisteswissenschaften
            "Zeitgenössische Kunstbewegungen", "Geschichte des Kinos", "Komponisten klassischer Musik",
            "Moderne Architekturstile", "Klassiker der Weltliteratur", "Philosophische Debatten",
            "Archäologische Entdeckungen", "Linguistische Theorien", "Mythologie und Folklore",
            "Darstellende Künste (Theater)", "Fotografietechniken", "Digitale Kunstschaffung",
            "Deutsche Literaturgeschichte", "Museumspädagogik", "Restaurierung historischer Artefakte",
            "Musikethnologie", "Filmtheorie", "Ästhetik",

            # Wirtschaft & Finanzen
            "Globales Lieferkettenmanagement", "Regulierung von Kryptowährungen", "Verhaltensökonomie",
            "Startup-Ökosysteme", "Internationale Handelsabkommen", "Soziale Unternehmensverantwortung (CSR)",
            "Marketingstrategien im digitalen Zeitalter", "Zukunft der Arbeit", "Modelle der Kreislaufwirtschaft",
            "Mikrofinanzinitiativen", "Volatilität an der Börse", "E-Commerce-Trends",
            "Auswirkungen der Globalisierung", "Steuerpolitik", "Wirtschaftsethik", "Insolvenzrecht",
            "Sharing Economy", "Industrie 4.0",

            # Gesundheit & Lebensstil
            "Bewusstsein für psychische Gesundheit", "Ernährungswissenschaftliche Forschung", "Alternative Heilmethoden",
            "Fitnesstechnologie", "Schlafforschung", "Präventivmedizin",
            "Psychologie des Glücks", "Stressbewältigungstechniken", "Digital Detox",
            "Achtsamkeit und Meditation", "Sucht-Hilfsprogramme", "Gesundes Altern",
            "Work-Life-Balance", "Patientenrechte", "Telemedizin", "Gesundheitssystemvergleich",
            "Impfforschung", "Sportmedizin",

            # Weltgeschehen & Geschichte
            "Geopolitische Konflikte", "Antike Zivilisationen", "Die Renaissance",
            "Geschichte der Weltkriege", "Postkoloniale Studien", "Theorien internationaler Beziehungen",
            "Spionage und Nachrichtendienste", "Revolutionen und soziale Umbrüche",
            "Historische Denkmäler und Stätten", "Diplomatie und Verhandlungen", "Ära des Kalten Krieges",
            "Geschichte der Europäischen Union", "Deutsche Wiedervereinigung", "Aufklärungsepoche",
            "Mittelalterliche Geschichte", "Industrielle Revolution", "Menschenrechtsgeschichte",

            # Natur & Umwelt
            "Erhaltung der Biodiversität", "Herausforderungen der Entwaldung", "Wildtierschutz",
            "Meeresbiologische Studien", "Geologische Formationen", "Extreme Wetterphänomene",
            "Projekte zur Wiederherstellung von Ökosystemen", "Gefährdete Arten", "Nationalparksysteme",
            "Städtische Grünflächen", "Maßnahmen zur Kontrolle der Umweltverschmutzung", "Vulkanologie",
            "Permakultur", "Wasserressourcenmanagement", "Lichtverschmutzung", "Bodenkunde",
            "Klimagerechtigkeit", "Umweltbildung",

            # Verschiedenes & Nischenthemen
            "Geschichte der Kochkunst", "Nachhaltigkeit in der Modebranche", "Wettbewerbsfähiges Gaming (E-Sport)",
            "Reise-Vlogging", "Heimwerken (DIY)", "Urban Gardening", "Brettspiel-Design",
            "Amateurfunkbetrieb", "Sammeln seltener Bücher", "Astrofotografie", "Ahnenforschung (Genealogie)",
            "Kaffee Kultur", "Minimalismus als Lebensstil", "Restaurierung von Oldtimern", "Podcast Produktion",
            "Bierbraukunst", "Kalligraphie"
        ]
        # Zufällig ein Thema auswählen
        return random.choice(topic_list)
       
    @traceable(run_type="llm")
    async def generate_match_title(self) -> ReadingMatchTitle:
        """Generates a new Reading Match Title on each call."""
        # Invoke the prompt template with current data
        prompt_data = prompt_template.invoke({
            "topic_list": self.get_topic_list()
        })
        
        # Generate the exam directly without caching
        exam = await self.llm.ainvoke(prompt_data)
        
        print(exam)
        
        return exam

