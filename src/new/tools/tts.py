import os
from pathlib import Path
from groq import Groq

client = Groq()

def speak(text, filename="speech.wav", voice="Aaliyah-PlayAI", model="playai-tts"):
    """
    Convert text to speech using Groq API and save as a wav file, then play it.
    """
    speech_file_path = Path(__file__).parent / filename
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        response_format="wav",
        input=text,
    )
    response.stream_to_file(speech_file_path)
    # Play the audio file (cross-platform)
    try:
        if os.name == 'nt':
            os.startfile(speech_file_path)
        elif os.name == 'posix':
            import subprocess
            subprocess.run(["aplay", str(speech_file_path)])
    except Exception as e:
        print(f"Audio file saved at {speech_file_path}, but could not be played automatically: {e}")
