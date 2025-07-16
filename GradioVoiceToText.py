import gradio as gr
from transformers import pipeline
import numpy as np


transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    try:
        result = transcriber({"sampling_rate": sr, "raw": y})
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Whisper STT") as demo:
    gr.Markdown("## 🎙️ English Speech-to-Text\nUpload or record your audio below to get a transcription.")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="🎧 Audio Input")
    
    with gr.Row():
        submit_btn = gr.Button("🔍 Transcribe")
        clear_btn = gr.Button("❌ Clear")
    
    text_output = gr.Textbox(label="📜 Transcription Output", lines=4, interactive=False)
    
    submit_btn.click(fn=transcribe, inputs=audio_input, outputs=text_output)
    clear_btn.click(fn=lambda: (None, ""), inputs=None, outputs=[audio_input, text_output])

demo.launch(share=True)


