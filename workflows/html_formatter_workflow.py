from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable

load_dotenv()

__all__ = ["HtmlFormatterWorkflow", "HtmlFormattedResult"]

class HtmlFormattedResult(BaseModel):
    formatted_text: str = Field(description="The formatted text with HTML tags")


class HtmlFormatterWorkflow:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1).with_structured_output(HtmlFormattedResult)

    def prepare_chain(self):
        prompt = PromptTemplate(
            template="""
            You are an expert HTML formatter. Take the following text and format it using only <span>, <br>, and <div> tags.
            You can use Tailwind CSS classes for styling if desired.
            The output MUST be a valid HTML snippet, without <html> or <body> tags.
            Return ONLY the formatted HTML text matching the required output schema.

            Input Text:
            {text}

            Formatted HTML Output:
            """
        )

        self.chain = prompt | self.llm
    
    @traceable(run_type="llm", name="format_html")
    async def format_html(self, text: str) -> HtmlFormattedResult:
        self.prepare_chain()
        return await self.chain.ainvoke({"text": text}) 
   