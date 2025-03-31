from dotenv import load_dotenv
from workflows.generate_transcript import generate_transcript, Conversation, Speaker
from elevenlabs import ElevenLabs, VoiceSettings
import os
import uuid
from itertools import cycle

# Load environment variables
load_dotenv()

elevenlabs_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# Default voice IDs from ElevenLabs
VOICE_IDS = {
    "male": [
        "kkJxCnlRCckmfFvzDW5Q",
        "wGD81UtSGECIRSWQjH8X",
        "FTNCalFNG5bRnkkaP5Ug",
        "9gSkuKCHRczfU5aLq1qU",
        "ekJ0doQ5Wa25P7W5HCj7",
    ],
    "female": [
        "7eVMgwCnXydb3CikjV7a",
        "mDRP1h6KfUD1XAUJxqr0",
        "wDvyXJwxWHsjOKSUVvpG",
        "Em6LomliZyRFu3w09pjD",
        "lzvBSKYbNWDD0a6BaJSK",
    ],
}


class ListeningExamService:
    def __init__(self):
        self.male_voices = cycle(VOICE_IDS["male"])
        self.female_voices = cycle(VOICE_IDS["female"])
        # Create audio directory if it doesn't exist
        os.makedirs("audio", exist_ok=True)

    def get_next_voice(self, gender: str) -> str:
        """Get the next available voice for the given gender."""
        if gender.lower() == "female":
            return next(self.female_voices)
        return next(self.male_voices)

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
            transcript: Conversation = generate_transcript(topic)

            for speaker in transcript.speakers:
                # Get next voice for the speaker's gender
                voice_id = self.get_next_voice(speaker.gender)

                # Get the audio stream
                audio_stream = elevenlabs_client.text_to_speech.convert(
                    text=speaker.opinion,
                    voice_id=voice_id,
                    output_format="mp3_22050_32",
                    model_id="eleven_flash_v2_5",
                    # Optional voice settings that allow you to customize the output
                    voice_settings=VoiceSettings(
                        stability=0.0,
                        similarity_boost=1.0,
                        style=0.0,
                        use_speaker_boost=True,
                        speed=1.0,
                    ),
                )

                # Convert generator to bytes
                audio_bytes = b"".join(list(audio_stream))

                unique_filename = f"audio/{uuid.uuid4()}.mp3"
                ## Save the audio to a file
                with open(unique_filename, "wb") as f:
                    f.write(audio_bytes)
                print(f"Saved audio for {speaker.name} as {unique_filename}")

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
