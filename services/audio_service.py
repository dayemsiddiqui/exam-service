from elevenlabs import ElevenLabs, VoiceSettings
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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


class AudioService:
    def __init__(self):
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
        # Create audio directory if it doesn't exist
        os.makedirs("audio", exist_ok=True)

    def get_voice(self, gender: str, speaker_index: int) -> str:
        """Get the voice for the given gender and speaker index."""
        voices = VOICE_IDS[gender.lower() if gender.lower() in VOICE_IDS else "male"]
        return voices[speaker_index % len(voices)]

    def generate_audio(self, text: str, gender: str, speaker_index: int) -> str:
        """
        Generate audio for the given text using a voice based on gender and speaker index.

        Args:
            text: The text to convert to speech
            gender: The gender of the speaker ("male" or "female")
            speaker_index: The index of the speaker in the conversation

        Returns:
            The filename of the generated audio file
        """
        try:
            # Get voice based on speaker's gender and position
            voice_id = self.get_voice(gender, speaker_index)

            # Get the audio stream
            audio_stream = self.client.text_to_speech.convert(
                text=text,
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

            # Generate unique filename
            filename = f"audio/{uuid.uuid4()}.mp3"

            # Save the audio to a file
            with open(filename, "wb") as f:
                f.write(audio_bytes)

            return filename
        except Exception as e:
            print(f"Audio generation error: {e}")
            raise
