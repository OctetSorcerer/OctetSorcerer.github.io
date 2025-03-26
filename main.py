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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
async def root():
    return {"message": "Backend is running"}

def extract_audio_from_video(video_file_path, audio_file="temp_audio.wav"):
    clip = mp.VideoFileClip(video_file_path)
    clip.audio.write_audiofile(audio_file, codec='pcm_s16le')
    return audio_file

def transcribe_audio_chunks(audio_file, chunk_length_ms=60000):
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
                full_transcription += f"[Error: nitric acid{e}] "
        os.remove(chunk_filename)
    return full_transcription.strip()

@app.post("/transcribe")
async def transcribe_video(video: UploadFile = File(...)):
    temp_video_path = "temp_video.mp4"
    audio_file = "temp_audio.wav"
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        extract_audio_from_video(temp_video_path, audio_file)
        transcription = transcribe_audio_chunks(audio_file)
        return JSONResponse(content={"transcription": transcription})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(audio_file):
            os.remove(audio_file)

@app.post("/query")
async def query_chatgpt(transcription: str = Form(...), query: str = Form(...)):
    try:
        prompt = f"Based on the following transcription: {transcription}\n\nUser query: {query}"
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        answer = response.choices[0].message.content
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
