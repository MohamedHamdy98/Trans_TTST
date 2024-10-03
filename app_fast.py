import os
import time
import uuid
import subprocess
import stat
import torch
import torchaudio
from zipfile import ZipFile
from pydub import AudioSegment
import re
import json
import langid
from deep_translator import GoogleTranslator
from speech_recognition import Recognizer, AudioFile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, Body, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
from pydantic import BaseModel

from model_setup import download_file

# Initialize FastAPI app
app = FastAPI()

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recognizer for speech recognition
recognizer = Recognizer()

# Directory for storing audio files

FINAL_OUTPUT_AUIDO = Path("/translation/audio_outputs/output_translated.wav")
FINAL_OUTPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)

# Download and set up ffmpeg binary
zip_file_path = download_file("https://huggingface.co/spaces/coqui/xtts/resolve/main/ffmpeg.zip", "./ffmpeg.zip")
binary_name = Path("ffmpeg")
if not binary_name.is_file():
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall()
    st = os.stat(binary_name)
    os.chmod(binary_name, st.st_mode | stat.S_IEXEC)

# Download and load Coqui XTTS V2 model
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(model_name)
model_path = Path(get_user_data_dir("tts")) / model_name.replace("/", "--")

config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=model_path / "model.pth",
    vocab_path=model_path / "vocab.json",
    eval=True,
    use_deepspeed=True,
)
model.cuda()

# Supported languages for text-to-speech and translation
supported_languages = config.languages
langs_for_audio = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'ja']
langs_for_translation = ['ar', 'zh-CN', 'en', 'fr', 'de', 'it', 'ja', 'nl', 'pt', 'pl', 'tr', 'ru', 'cs', 'es']
MAX_TEXT_LENGTH = {
    'en': 1000, 'es': 1000, 'fr': 1000, 'de': 1000, 'it': 1000,
    'pt': 1000, 'pl': 1000, 'tr': 1000, 'ru': 1000, 'nl': 1000,
    'cs': 1000, 'ar': 500, 'zh-cn': 500, 'ja': 500
}

class TranslationRequest(BaseModel):
    prompt: str
    language: str
    trans_lang: str
    audio_file_pth: Path = None
    mic_file_path: Path = None
    use_mic: bool = False
    voice_cleanup: bool = False
    no_lang_auto_detect: bool = False

def transcribe_audio(audio_file, language, lang_auto_detect):
    try:
        with AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=language or lang_auto_detect)
            return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def translate_text(text, src_lang, trans_lang, lang_auto_detect):
    try:
        translator = GoogleTranslator(source=src_lang or lang_auto_detect, target=trans_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Error translating text: {e}")
        return None

def predict(prompt, language, audio_file_pth=None, mic_file_path=None, use_mic=False, voice_cleanup=False, no_lang_auto_detect=False):
    if language not in supported_languages:
        print(f"Language {language} is not supported.")
        return None

    language_predicted = langid.classify(prompt)[0].strip()
    if language_predicted == "zh":
        language_predicted = "zh-cn"

    max_length = MAX_TEXT_LENGTH.get(language, 200)
    chunks = [prompt[i:i + max_length] for i in range(0, len(prompt), max_length)] if len(prompt) > max_length else [prompt]

    speaker_wav = mic_file_path if use_mic else audio_file_pth

    if use_mic and not mic_file_path:
        print("Please record your voice with a Microphone.")
        return None

    if voice_cleanup:
        out_filename = Path("/translation/voice_cleanup/output_translated.wav")
        shell_command = f"ffmpeg -y -i {speaker_wav} -af lowpass=8000,highpass=75,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02 {out_filename}"
        subprocess.run(shell_command, shell=True, capture_output=False, text=True, check=True)
        speaker_wav = out_filename

    all_outputs = []
    for chunk in chunks:
        try:
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60
            )
            out = model.inference(
                chunk, language, gpt_cond_latent, speaker_embedding,
                repetition_penalty=5.0, temperature=0.75
            )
            if out is None or "wav" not in out or out["wav"] is None:
                print("No audio data returned.")
                continue
            output_path = Path(f"/translation/bf_merged/output_{uuid.uuid4()}.wav")
            torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
            all_outputs.append(output_path)
        except RuntimeError as e:
            return None

    combined = AudioSegment.empty()
    for audio_file in all_outputs:
        audio = AudioSegment.from_wav(audio_file)
        combined += audio

    output_combined_path = FINAL_OUTPUT_AUIDO
    combined.export(output_combined_path, format="wav")
    return output_combined_path

@app.post("/translate")
async def translate(request: Request, audio_file_pth: UploadFile = File(None), mic_file_path: UploadFile = File(None)):
    # Determine if the request is JSON or form data
    if request.headers.get("content-type") == "application/json":
        data = await request.json()
        prompt = data.get("prompt")
        language = data.get("language")
        trans_lang = data.get("trans_lang")
        use_mic = data.get("use_mic", False)
        voice_cleanup = data.get("voice_cleanup", False)
        no_lang_auto_detect = data.get("no_lang_auto_detect", False)
    else:
        prompt = await request.form()
        language = prompt.get("language")
        trans_lang = prompt.get("trans_lang")
        use_mic = prompt.get("use_mic", False)
        voice_cleanup = prompt.get("voice_cleanup", False)
        no_lang_auto_detect = prompt.get("no_lang_auto_detect", False)

    if not prompt or not language or not trans_lang:
        raise HTTPException(status_code=400, detail="Missing required parameters.")

    if not use_mic and audio_file_pth is None:
        raise HTTPException(status_code=400, detail="Audio file path is required when not using a microphone.")

    # Handle file upload if provided
    if audio_file_pth:
        audio_file_pth = Path(f"/tmp/{audio_file_pth.filename}")
        with open(audio_file_pth, "wb") as buffer:
            buffer.write(await audio_file_pth.read())

    transcribed_text = transcribe_audio(audio_file_pth or mic_file_path, language, no_lang_auto_detect) if not prompt else prompt
    translated_text = translate_text(transcribed_text, language, trans_lang, no_lang_auto_detect)

    if translated_text is None:
        return JSONResponse(content={"error": "Translation failed."})

    output_audio_path = predict(translated_text, trans_lang, audio_file_pth, mic_file_path, use_mic, voice_cleanup, no_lang_auto_detect)
    if output_audio_path is None:
        return JSONResponse(content={"error": "Audio generation failed."})

    return JSONResponse(content={"file_path": str(output_audio_path)})

@app.get("/get_path_translated")
async def get_path_translated():
    output_path = FINAL_OUTPUT_AUDIO
    if output_path.is_file():
        return JSONResponse(content={"file_path": str(output_path)})
    raise HTTPException(status_code=404, detail="File not found.")

@app.get("/")
async def index():
    return {"message": "Running 😀!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 
