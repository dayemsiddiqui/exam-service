import os
import random
import io
from openai import OpenAI
from dotenv import load_dotenv
from workflows.generate_interview import ConversationSegment
import instructor
from typing import List, Generator, Optional
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Define consistent voices
OPENAI_VOICES = {
    "male": "onyx",
    "female": "nova",
}

# Define silence duration range in milliseconds
MIN_SILENCE_MS = 300
MAX_SILENCE_MS = 700

class AudioListeningInterviewExamService:
    def __init__(self, max_workers: Optional[int] = None):
        self.client = instructor.patch(OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        ))
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        # Initialize ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def get_voice(self, gender: str) -> str:
        """Get the consistent OpenAI voice for the given gender."""
        return OPENAI_VOICES.get(gender.lower(), OPENAI_VOICES["male"])

    def _generate_segment_audio_segment(self, segment: ConversationSegment) -> Optional[AudioSegment]:
        """Generates audio, loads it into pydub AudioSegment, returns it or None on error."""
        try:
            voice = self.get_voice(segment.speaker_gender)
            response = self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=segment.text,
                response_format="mp3"
            )
            audio_bytes = response.read()
            # Load bytes into pydub segment
            segment_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            return segment_audio
        except Exception as e:
            # Catch OpenAI errors or pydub loading errors
            print(f"[Error] Failed processing segment '{segment.text[:30]}...': {e}")
            return None # Indicate failure

    def _generate_silence_segment(self, duration_ms: int) -> AudioSegment:
        """Generates silent pydub AudioSegment."""
        return AudioSegment.silent(duration=duration_ms)

    def generate_concatenated_streaming_audio(self, segments: List[ConversationSegment]) -> Generator[bytes, None, None]:
        """
        Generates audio segments in parallel, concatenates them using pydub 
        with variable silence, exports the final audio, and streams the result.

        Args:
            segments: A list of ConversationSegment objects.

        Yields:
            Bytes chunks of the final concatenated MP3 audio stream.
        """
        CHUNK_SIZE = 4096
        # Submit tasks to generate AudioSegment objects
        futures = [self.executor.submit(self._generate_segment_audio_segment, segment) for segment in segments]

        # Initialize an empty AudioSegment to accumulate results
        final_audio = AudioSegment.empty()
        is_first_successful_part = True

        # Process futures and concatenate AudioSegments
        for future in futures:
            try:
                segment_audio = future.result() # Get the AudioSegment result

                if segment_audio:
                    # Add silence before this segment if it's not the very first successful part
                    if not is_first_successful_part:
                        silence_duration = random.randint(MIN_SILENCE_MS, MAX_SILENCE_MS)
                        silence_segment = self._generate_silence_segment(silence_duration)
                        final_audio += silence_segment # Concatenate using pydub
                    else:
                        is_first_successful_part = False

                    # Concatenate the actual segment audio
                    final_audio += segment_audio
                else:
                    print(f"[Warning] Skipping segment due to generation failure.")

            except Exception as e:
                print(f"[Error] Failed to retrieve or process result from future: {e}")

        # Export the final combined audio to an in-memory buffer
        final_buffer = io.BytesIO()
        try:
            if len(final_audio) > 0:
                 final_audio.export(final_buffer, format="mp3")
                 final_buffer.seek(0) # Rewind buffer to the beginning for reading
            else:
                 print("[Warning] No audio segments were successfully generated. Returning empty stream.")
                 # final_buffer remains empty
        except Exception as e:
            print(f"[Error] Failed to export final concatenated audio: {e}")
            # Handle export error, maybe yield nothing or raise
            final_buffer = io.BytesIO() # Ensure it's an empty buffer on error
        
        # --- Generator function to stream the final bytes from the buffer --- 
        def _stream_final_audio():
            while True:
                chunk = final_buffer.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        # --- End of generator function ---

        return _stream_final_audio()

    def shutdown_executor(self):
        """Should be called on application shutdown to clean up the executor."""
        self.executor.shutdown(wait=True)

# Consider adding executor shutdown logic to your FastAPI app's shutdown event
# Example (in main.py):
# @app.on_event("shutdown")
# def shutdown_event():
#     audio_listening_interview_exam_service.shutdown_executor()

# Example usage (optional, for testing)
# if __name__ == "__main__":
#    service = AudioListeningInterviewExamService()
#    example_segments = [
#        ConversationSegment(speaker="interviewer", text="Hallo, willkommen.", speaker_gender="female"),
#        ConversationSegment(speaker="interviewee", text="Danke.", speaker_gender="male"),
#        ConversationSegment(speaker="interviewer", text="Wie geht es Ihnen?", speaker_gender="female"),
#    ]
#    audio_stream_generator = service.generate_concatenated_streaming_audio(example_segments)
#
#    output_filename = "test_parallel_output.mp3"
#    with open(output_filename, "wb") as f:
#        for chunk in audio_stream_generator:
#            f.write(chunk)
#    print(f"Audio saved to {output_filename}")
#    service.shutdown_executor() # Shutdown executor after use 