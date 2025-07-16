import gradio as gr
from transformers import pipeline
import numpy as np

# Load the Whisper model for English speech recognition
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# Define the transcription function
def transcribe(audio):
    sr, y = audio
    if y.ndim > 1:
        y = y.mean(axis=1)  # Convert stereo to mono
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))  # Normalize audio
    return transcriber({"sampling_rate": sr, "raw": y})["text"]

# Create the Gradio interface
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="microphone", type="numpy"),
    outputs="text",
    title="English Speech-to-Text",
    description="Speak into your mic and get the transcription in English."
)

demo.launch()

