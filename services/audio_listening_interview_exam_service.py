import os
from openai import OpenAI
from dotenv import load_dotenv
from workflows.generate_interview import ConversationSegment  # Assuming this path is correct
import instructor

# Load environment variables
load_dotenv()

# Define consistent voices
OPENAI_VOICES = {
    "male": "onyx",  # Example voice, choose preferred ones
    "female": "nova", # Example voice, choose preferred ones
}

class AudioListeningInterviewExamService:
    def __init__(self):
        # Patch the OpenAI client with instructor
        self.client = instructor.patch(OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        ))
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

    def get_voice(self, gender: str) -> str:
        """Get the consistent OpenAI voice for the given gender."""
        return OPENAI_VOICES.get(gender.lower(), OPENAI_VOICES["male"]) # Default to male if gender unknown

    def generate_streaming_audio(self, segment: ConversationSegment):
        """
        Generate streaming audio for the given conversation segment using OpenAI TTS.

        Args:
            segment: The ConversationSegment containing text and speaker gender.

        Returns:
            A streaming audio response.
        """
        try:
            voice = self.get_voice(segment.speaker_gender)

            response = self.client.audio.speech.create(
                model="tts-1", # Or "tts-1-hd"
                voice=voice,
                input=segment.text,
                response_format="mp3", # Or other supported formats like opus, aac, flac
                # speed=1.0 # Optional: Adjust speed if needed
            )

            # Stream the audio data
            return response.iter_bytes(chunk_size=4096)

        except Exception as e:
            print(f"OpenAI audio generation error: {e}")
            # Re-raise or handle as appropriate for your error strategy
            raise

# Example usage (optional, for testing)
# if __name__ == "__main__":
#     service = AudioListeningInterviewExamService()
#     example_segment = ConversationSegment(
#         speaker="interviewer",
#         text="Hallo, wie geht es Ihnen heute?",
#         speaker_gender="female"
#     )
#     audio_stream = service.generate_streaming_audio(example_segment)
#
#     # Example of how to save the stream to a file
#     output_filename = "test_output.mp3"
#     with open(output_filename, "wb") as f:
#         for chunk in audio_stream:
#             f.write(chunk)
#     print(f"Audio saved to {output_filename}") 