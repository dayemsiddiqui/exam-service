from dotenv import load_dotenv
from workflows.generate_transcript import (
    generate_listening_exam_transcript,
    Conversation,
    Speaker,
)
from services.audio_service import AudioService

# Load environment variables
load_dotenv()


class ListeningExamService:
    def __init__(self):
        self.audio_service = AudioService()

    def generate_transcript(
        self, topic: str = "Is friendship important to you?"
    ) -> Conversation:
        """
        Generates a listening exam transcript with questions and answers.

        Args:
            topic: The topic for the conversation

        Returns:
            A Conversation object containing the dialogue, questions, and answers
        """
        try:
            transcript: Conversation = generate_listening_exam_transcript(topic)
            return transcript
        except Exception as e:
            print(f"Transcript generation error: {e}")
            # Return a simple fallback conversation with the correct structure
            return Conversation(
                speakers=[
                    Speaker(
                        name="System",
                        gender="none",
                        opinion="Error generating conversation",
                        question="Error generating question?",
                        correct_answer=False,
                    )
                ]
            )

    def generate_conversation(
        self, topic: str = "Is friendship important to you?"
    ) -> Conversation:
        """
        Generates a listening exam conversation with questions and answers.

        Args:
            topic: The topic for the conversation

        Returns:
            A Conversation object containing the dialogue, questions, and answers
        """
        try:
            transcript: Conversation = generate_listening_exam_transcript(topic)

            for idx, speaker in enumerate(transcript.speakers):
                # Generate audio for each speaker
                audio_file = self.audio_service.generate_audio(
                    text=speaker.opinion, gender=speaker.gender, speaker_index=idx
                )
                print(
                    f"Saved audio for {speaker.name} ({speaker.gender}) as {audio_file}"
                )

            return transcript
        except Exception as e:
            print(f"Conversation generation error: {e}")
            # Return a simple fallback conversation with the correct structure
            return Conversation(
                speakers=[
                    Speaker(
                        name="System",
                        gender="none",
                        opinion="Error generating conversation",
                        question="Error generating question?",
                        correct_answer=False,
                    )
                ]
            )
