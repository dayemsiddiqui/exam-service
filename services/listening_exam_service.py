from dotenv import load_dotenv
from workflows.generate_transcript import (
    generate_listening_exam_transcript,
    Conversation,
    Speaker,
)
from services.audio_service import AudioService
from itertools import cycle

# Load environment variables
load_dotenv()

# Define a list of diverse B1-level topics
DEFAULT_TOPICS = [
    "What are your opinions on home schooling?",
    "What do you think about electric cars?",
    "What are your opinions on the impact of social media?",
    "What are your opinions on the impact of AI on society?",
    "Is friendship important to you?",
    "Is it better to be single or in a relationship?",
    "What are your opinions on capitalism",
    "What do you think about the impact of globalization?",
    "What do you think about the impact of the internet on our lives?",
    "What do you think about the impact of smartphones on our lives?",
    "Is it better to live in a big city or a small town?",
    "Is traveling alone better than traveling with a group?",
    "Is it better to take a job you don't like or to start your own business?",
    "What do you think about the impact of the internet on our lives?",
    "Is money important to you or not?",
    "Do you prefer hiking or swimming?",
    "Homecooking or eating out, what do you prefer?",
]


class ListeningExamService:
    def __init__(self):
        self.audio_service = AudioService()
        self._topic_cycle = cycle(DEFAULT_TOPICS)

    def get_next_topic(self) -> str:
        """
        Get the next topic from the round-robin cycle.

        Returns:
            str: The next topic from the cycle
        """
        return next(self._topic_cycle)

    def generate_transcript(self, topic: str = None) -> Conversation:
        """
        Generates a listening exam transcript with questions and answers.

        Args:
            topic: The topic for the conversation. If None, uses round-robin selection.

        Returns:
            A Conversation object containing the dialogue, questions, and answers
        """
        try:
            # If no topic provided, get the next topic from the cycle
            if topic is None:
                topic = self.get_next_topic()

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
