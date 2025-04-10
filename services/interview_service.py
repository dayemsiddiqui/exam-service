from workflows.generate_interview import generate_interview_transcript, Interview
from langsmith import traceable
from typing import List, Optional
import random

class InterviewService:
    """Service to handle the generation of interview transcripts for listening exams."""

    def __init__(self):
        """Initializes the InterviewService."""
        pass # No initialization needed

    @traceable(run_type="chain")
    def generate_interview(self) -> Interview:
        """Generates an interview transcript focused on the interviewee's life/career.

        Returns:
            An Interview object containing the generated content.
        """
        # Call the generation function from the workflow
        try:
            # The workflow now focuses the questions based on the interviewee's profile
            interview_data = generate_interview_transcript()
            return interview_data
        except Exception as e:
            print(f"Error generating interview transcript: {e}")
            # Re-raise with more context
            raise Exception(f"Failed to generate interview: {str(e)}") 