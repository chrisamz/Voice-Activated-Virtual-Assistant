# speech_recognition.py

"""
Speech Recognition Module for Voice-Activated Virtual Assistant

This module contains functions for converting spoken language into text using
speech recognition models.

Techniques Used:
- Acoustic modeling
- Language modeling
- Feature extraction

Libraries/Tools:
- Google Speech-to-Text
- CMU Sphinx
- DeepSpeech
"""

import speech_recognition as sr
import os

class SpeechRecognition:
    def __init__(self, model='google'):
        """
        Initialize the SpeechRecognition class.
        
        :param model: str, speech recognition model to use ('google', 'sphinx', 'deepspeech')
        """
        self.model = model
        self.recognizer = sr.Recognizer()
        
    def recognize_speech(self, audio_file):
        """
        Recognize speech from an audio file.
        
        :param audio_file: str, path to the audio file
        :return: str, recognized text
        """
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
            
            try:
                if self.model == 'google':
                    text = self.recognizer.recognize_google(audio)
                elif self.model == 'sphinx':
                    text = self.recognizer.recognize_sphinx(audio)
                elif self.model == 'deepspeech':
                    text = self.recognizer.recognize_deepspeech(audio)
                else:
                    raise ValueError(f"Model {self.model} not supported.")
                return text
            except sr.UnknownValueError:
                return "Speech Recognition could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results from Speech Recognition service; {e}"

if __name__ == "__main__":
    # Example usage
    audio_file = "path/to/your/audio/file.wav"
    
    # Initialize the speech recognition system
    sr_system = SpeechRecognition(model='google')
    
    # Recognize speech from the audio file
    recognized_text = sr_system.recognize_speech(audio_file)
    print("Recognized Text:", recognized_text)
