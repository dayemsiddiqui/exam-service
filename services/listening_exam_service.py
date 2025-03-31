from dotenv import load_dotenv
from workflows.generate_transcript import generate_transcript, Conversation

# Load environment variables
load_dotenv()


class ListeningExamService:
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
            transcript = generate_transcript(topic)

            print(transcript)
            return transcript
        except Exception as e:
            print(f"Conversation generation error: {e}")
            # Return a simple fallback conversation
            return Conversation(
                context="Error generating conversation",
                dialogue=[
                    {"speaker": "System", "text": "Error generating conversation"}
                ],
                questions=[{"question": "Error generating questions"}],
                answers=[{"answer": "Error generating answers"}],
            )
