import speech_recognition as sr
from faster_whisper import WhisperModel
from colorama import Fore, Style
import time
import os

from graph import chat_bot
from voice_service import play_text_to_speech

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Whisper model
model = WhisperModel('medium.en', device="cpu", compute_type="int8")

# Initialize SpeechRecognizer
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Full transcript variable (Accumulates transcription across all iterations)
full_transcript = ""

hallucinated_phrases = ["thank you.", "thank you very much.", "hmm", "hmmm"]

question1 = "Why do we consider blood as a connective tissue?"
question2 = '''Estimate the fraction of molecular volume to the actual volume occupied by oxygen
gas at STP. Take the diameter of an oxygen molecule to be 3 Å.'''
question3 = "How many years dinosourous were there in the planet Earth?"
question4 = "Briefly explain me about molecular nature of matter for a 3 mark question."
question5 = "What is kinetic theory of an ideal gas?"
question6 = "Hi, WassUp ?"

print("Listening... Speak into the microphone.")

try:
    last_speech_time = time.time()  # Track the last time speech was detected
    mybot=chat_bot()
    workflow=mybot()

    config = {"configurable": {"user_id": "102",
                        # Checkpoints are accessed by thread_id
                            "thread_id": "1",
                                }
                }
    while True:
        with microphone as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)

            print(Fore.LIGHTYELLOW_EX + "Listening... Speak your sentence." + Style.RESET_ALL)
            try:
                # Listen for audio
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=None)
            except sr.WaitTimeoutError:
                # If no speech is detected within 5 seconds
                if time.time() - last_speech_time > 5:
                    print("No speech detected for 5 seconds. Exiting...")
                    break
                else:
                    continue

        # Update the time of the last speech detection
        last_speech_time = time.time()

        # Save audio to a temporary WAV file
        with open("temp_audio.wav", "wb") as temp_file:
            temp_file.write(audio.get_wav_data())

        # Transcribe using faster-whisper
        segments, _ = model.transcribe("temp_audio.wav")
        curr_text = ""
        for segment in segments:
            curr_text += segment.text.strip()

        # Append to the full transcript
        full_transcript += curr_text + " "
        print(Fore.LIGHTRED_EX + "Human : " + Style.RESET_ALL, end="")
        print(Fore.BLUE + f"{curr_text}" + Style.RESET_ALL)

        # Check for "exit" command
        # if "ok bye" in curr_text.lower() or "okay bye" in curr_text.lower():
        #     print(Fore.RED + "Exit command detected. Stopping..." + Style.RESET_ALL)
        #     break

        if curr_text == "" or curr_text.lower() in hallucinated_phrases:
            print(Fore.YELLOW + "Silence detected. Exiting..." + Style.RESET_ALL)
            break

        inputs = {
                    "messages": curr_text
                }
        
        response=workflow.invoke(inputs, config=config)
        # print('-'*120)
        output = None
        # if 'generation' not in response:
        #     output = response['messages'][-1].content
            
        # else:
        #     output = response['generation']
        output = response['messages'][-1].content

        print(Fore.LIGHTRED_EX + "AI : " + Style.RESET_ALL, end="")
        print(Fore.GREEN + output + Style.RESET_ALL)
        play_text_to_speech(output)
        print('-'*120)

        if "ok bye" in curr_text.lower() or "okay bye" in curr_text.lower():
            print(Fore.RED + "Exit command detected. Stopping..." + Style.RESET_ALL)
            break


except KeyboardInterrupt:
    print("\nStopped listening...")

os.remove("temp_audio.wav")
# Final Output
# print("Full Transcript: " + Fore.MAGENTA + f"{full_transcript}" + Style.RESET_ALL)
