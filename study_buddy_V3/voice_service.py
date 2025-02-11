import os
import sys
import time
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
# Suppress Pygame output
def suppress_pygame_output():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def restore_output():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Suppress Pygame initialization output
suppress_pygame_output()
import pygame
restore_output()

def play_text_to_speech(text, language='en', slow=False):
    tts = gTTS(text=text, lang=language, slow=slow)
    
    temp_audio_file = "temp_audio_output.mp3"
    tts.save(temp_audio_file)
    
    # Load and adjust the playback speed using pydub
    audio = AudioSegment.from_file(temp_audio_file)
    faster_audio = audio.speedup(playback_speed=1.2)
    
    # Play the adjusted audio
    play(faster_audio)
    
    # Clean up temporary file
    os.remove(temp_audio_file)

# play_text_to_speech("What's up bro ?, how you doing ? This is your boy Asit !")