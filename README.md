<h1 align="center">🎙️ Whisper Speech-to-Text App</h1>

<p align="center">
  A real-time English speech-to-text app powered by <strong>OpenAI Whisper</strong> and <strong>Gradio</strong>.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/sherintht/whisper-stt" target="_blank">
    <img src="https://img.shields.io/badge/Live Demo-Click Here-success?style=flat&logo=gradio" alt="Live Demo">
  </a>
  <a href="https://github.com/sherintht/whisper-stt" target="_blank">
    <img src="https://img.shields.io/github/stars/sherintht/whisper-stt?style=social" alt="GitHub stars">
  </a>
</p>

<hr>

<h2>🚀 Features</h2>

<ul>
  <li>🎤 Record directly from your microphone or upload audio files</li>
  <li>⚡ Fast and lightweight using <code>openai/whisper-base.en</code></li>
  <li>📜 Clean text output of the transcription</li>
  <li>🧪 Powered by Hugging Face Transformers and Gradio UI</li>
</ul>

<h2>🛠️ Technologies Used</h2>

<ul>
  <li><strong>Python</strong></li>
  <li><strong>Gradio</strong> for web UI</li>
  <li><strong>Transformers</strong> (Hugging Face) for model inference</li>
  <li><strong>Torch</strong> and <strong>Torchaudio</strong> for audio processing</li>
</ul>

<h2>📦 Installation</h2>

<ol>
  <li>Clone the repository:</li>

<pre><code>git clone https://github.com/sherintht/whisper-stt.git
cd whisper-stt
</code></pre>

  <li>Create a virtual environment (optional but recommended):</li>

<pre><code>python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
</code></pre>

  <li>Install the required packages:</li>

<pre><code>pip install -r requirements.txt</code></pre>

  <li>Run the app locally:</li>

<pre><code>python app.py</code></pre>

</ol>

<h2>🌐 Live Demo</h2>

<p>
Check out the deployed app on Hugging Face Spaces:<br>
👉 <a href="https://sherintht-whisper-stt.hf.space" target="_blank">https://sherintht-whisper-stt.hf.space</a>
</p>

<h2>📜 License</h2>
<p>This project is licensed under the <strong>MIT License</strong>.</p>
