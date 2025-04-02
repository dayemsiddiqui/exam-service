from dotenv import load_dotenv
from workflows.generate_announcements import (
    generate_listening_exam_announcement,
    Announcement,
    Announcer,
)
from services.audio_service import AudioService

# Load environment variables
load_dotenv()




class ListeningExamAnnouncementService:
    def __init__(self):
        self.audio_service = AudioService()


    def generate_announcement(self) -> Announcement:
        """
        Generates a listening exam announcement with questions and answers.

        Returns:
            A Conversation object containing the announcement, questions, and answers
        """
        try:

            # Generate the announcement
            announcement: Announcement = generate_listening_exam_announcement()
            return announcement
        except Exception as e:
            print(f"Announcement generation error: {e}")
            # Return a simple fallback conversation with the correct structure
            return Announcement(
                speakers=[
                    Announcer(
                        name="Announcer",
                        gender="female",
                        opinion="Error generating announcement",
                        question="Error generating question?",
                        correct_answer=False,
                        explanation="Error occurred during generation",
                        english_translation="Error generating announcement",
                    )
                ]
            )
