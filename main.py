# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
from openai import OpenAI

app = FastAPI()

# Allow CORS so the frontend on GitHub Pages can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your GitHub Pages URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client with your API key (set as an environment variable)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_audio_from_video(video_file_path, audio_file="temp_audio.wav"):
    """Extracts audio from the video and saves it as a WAV file."""
    clip = mp.VideoFileClip(video_file_path)
    clip.audio.write_audiofile(audio_file, codec='pcm_s16le')
    return audio_file

def transcribe_audio_chunks(audio_file, chunk_length_ms=60000):
    """Splits audio into chunks and transcribes each chunk."""
    sound = AudioSegment.from_wav(audio_file)
    duration_ms = len(sound)
    recognizer = sr.Recognizer()
    full_transcription = ""

    for i in range(0, duration_ms, chunk_length_ms):
        chunk = sound[i:i + chunk_length_ms]
        chunk_filename = f"chunk{i}.wav"
        chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                full_transcription += text + " "
            except sr.UnknownValueError:
                full_transcription += "[Unrecognized speech] "
            except sr.RequestError as e:
                full_transcription += f"[Error: {e}] "
        os.remove(chunk_filename)

    return full_transcription.strip()

@app.post("/transcribe")
async def transcribe_video(video: UploadFile = File(...)):
    """Endpoint to receive a video, transcribe it, and return the transcription."""
    # Save the uploaded video temporarily
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Extract and transcribe audio
    audio_file = extract_audio_from_video(temp_video_path)
    transcription = transcribe_audio_chunks(audio_file)

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(audio_file)

    return JSONResponse(content={"transcription": transcription})

@app.post("/query")
async def query_chatgpt(transcription: str = Form(...), query: str = Form(...)):
    """Endpoint to send transcription and user query to ChatGPT."""
    # Prepare the prompt with transcription as context
    prompt = f"Based on the following transcription: {transcription}\n\nUser query: {query}"
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500  # Adjust as needed
    )
    answer = response.choices[0].message.content
    return JSONResponse(content={"answer": answer})