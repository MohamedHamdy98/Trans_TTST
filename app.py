from model_setup import setup_environment, download_file

# Initialize the Roop model setup
# setup_environment()
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from flask import Flask, request, jsonify, send_file

import uuid
import subprocess
from zipfile import ZipFile
import stat
import torch
import torchaudio
from pydub import AudioSegment
import re
# from IPython.display import Audio
import langid
from flask_cors import CORS
from deep_translator import GoogleTranslator
from speech_recognition import Recognizer, AudioFile


from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir

# Initialize Flask application
app = Flask(__name__)
CORS(app)


def check_model_downloaded(model_name, model_path):
    """Check if the model files exist."""
    return os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "model.pth"))

# Check if GPU is available, otherwise use CPU
torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a directory to store generated audio files
AUDIO_STORAGE_DIR = '/tmp/translated/'
Final_OUTPUT_AUIDO = "/tmp/translated/output_translated.wav"

os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)
os.makedirs(Final_OUTPUT_AUIDO, exist_ok=True)

# Export and set executable permissions for a newer ffmpeg binary
# Path to the zip file and the binary within it
zip_file_path = download_file("https://huggingface.co/spaces/coqui/xtts/resolve/main/ffmpeg.zip", "./ffmpeg.zip")
# zip_file_path = r"H:\avatar_veem\Edit_Coding\With_Docker\Trans\xtts2-hf\ffmpeg.zip"
binary_name = "ffmpeg"

# Check if the binary already exists
if os.path.isfile(binary_name):
    print(f"{binary_name} already exists and is executable.")
else:
    # Extract the zip file
    try:
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall()
        print(f"Successfully extracted {zip_file_path}")
    except Exception as e:
        print(f"Failed to extract {zip_file_path}: {e}")
        raise

    # Make the ffmpeg binary executable
    try:
        if os.path.isfile(binary_name):
            # Get current permissions
            st = os.stat(binary_name)
            # Set executable permission
            os.chmod(binary_name, st.st_mode | stat.S_IEXEC)
            print(f"Successfully made {binary_name} executable")
        else:
            print(f"{binary_name} does not exist after extraction")
    except Exception as e:
        print(f"Failed to set executable permissions for {binary_name}: {e}")
        raise

# Initialize recognizer for speech recognition
recognizer = Recognizer()


# Download and load Coqui XTTS V2 model
print("Downloading if not downloaded Coqui XTTS V2")
from TTS.utils.manage import ModelManager

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
print("XTTS downloaded")

config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
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

def transcribe_audio(audio_file, language, lang_auto_detect):
    """
    Transcribe audio from a file into text using Google Speech Recognition.

    :param audio_file: Path to the audio file.
    :param language: Language code for speech recognition. If None, auto-detect language.
    :param lang_auto_detect: Language auto-detection setting.
    :return: Transcribed text or None if an error occurs.
    """
    try:
        with AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=language or lang_auto_detect)
            return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def translate_text(text, src_lang, trans_lang, lang_auto_detect):
    """
    Translate text from source language to target language using Google Translator.

    :param text: Text to be translated.
    :param src_lang: Source language code. If None, auto-detect language.
    :param trans_lang: Target language code.
    :param lang_auto_detect: Language auto-detection setting.
    :return: Translated text or None if an error occurs.
    """
    try:
        translator = GoogleTranslator(source=src_lang or lang_auto_detect, target=trans_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Error translating text: {e}")
        return None

def predict(prompt, language, audio_file_pth=None, mic_file_path=None, use_mic=False, voice_cleanup=False, no_lang_auto_detect=False):
    """
    Generate speech from text using the XTTS model. Optionally, clean up audio and handle microphone input.

    :param prompt: Text to be converted to speech.
    :param language: Language code for text-to-speech.
    :param audio_file_pth: Path to reference audio file (if use_mic is False).
    :param mic_file_path: Path to microphone audio file (if use_mic is True).
    :param use_mic: Boolean flag to indicate whether to use microphone input.
    :param voice_cleanup: Boolean flag to indicate whether to clean up voice input.
    :param no_lang_auto_detect: Boolean flag to disable language auto-detection.
    :return: Path to the generated audio file or None if an error occurs.
    """
    if language not in supported_languages:
        print(f"Language {language} is not supported. Please choose from the supported languages.")
        return None

    language_predicted = langid.classify(prompt)[0].strip()
    if language_predicted == "zh":
        language_predicted = "zh-cn"

    print(f"Detected language: {language_predicted}, Chosen language: {language}")

    if len(prompt) > 15 and language_predicted != language and not no_lang_auto_detect:
        print("Detected language does not match chosen language. Please disable language auto-detection if you're sure.")
        return None

    max_length = MAX_TEXT_LENGTH.get(language, 200)
    chunks = [prompt[i:i + max_length] for i in range(0, len(prompt), max_length)] if len(prompt) > max_length else [prompt]

    speaker_wav = mic_file_path if use_mic else audio_file_pth

    if use_mic and not mic_file_path:
        print("Please record your voice with Microphone, or uncheck Use Microphone to use reference audios.")
        return None

    if voice_cleanup:
        try:
            out_filename = "/tmp/translated/voice_cleanup/output_translated.wav"
            shell_command = f"ffmpeg -y -i {speaker_wav} -af lowpass=8000,highpass=75,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02 {out_filename}"
            subprocess.run(shell_command, shell=True, capture_output=False, text=True, check=True)
            speaker_wav = out_filename
            print("Filtered microphone input")
        except subprocess.CalledProcessError as e:
            print("Error filtering audio:", e)
            return None

    metrics_text = ""
    all_outputs = []

    for chunk in chunks:
        try:
            if len(chunk) < 2:
                print("Chunk is too short.")
                continue

            print("Generating new audio...")
            t_latent = time.time()

            try:
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60
                )
            except Exception as e:
                print("Speaker encoding error:", str(e))
                return None

            latent_calculation_time = time.time() - t_latent
            print(f"Latent calculation time: {latent_calculation_time} seconds")

            chunk = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)", r"\1 \2\2", chunk)

            t0 = time.time()
            out = model.inference(
                chunk, language, gpt_cond_latent, speaker_embedding,
                repetition_penalty=5.0, temperature=0.75
            )
            inference_time = time.time() - t0
            print(f"Time to generate audio for chunk: {round(inference_time * 1000)} milliseconds")

            if out is None or "wav" not in out or out["wav"] is None:
                print("No audio data was returned from the model.")
                continue

            real_time_factor = (time.time() - t0) / out["wav"].shape[-1] * 24000
            print(f"Real-time factor (RTF): {real_time_factor}")

            output_path = f"/tmp/translated/bf_merged/output_{uuid.uuid4()}.wav"
            torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
            metrics_text += f"Chunk processed. Time to generate audio: {round(inference_time * 1000)} milliseconds\n"
            metrics_text += f"Real-time factor (RTF): {real_time_factor:.2f}\n"
            all_outputs.append(output_path)

        except RuntimeError as e:
            if "device-side assert" in str(e):
                print(f"Unrecoverable exception caused by language:{language} prompt:{chunk}")
                print("Unhandled Exception. Please retry.")
            else:
                print("RuntimeError:", str(e))
                print("An unexpected error occurred. Please try again.")
            return None

    # Merge all audio files into one
    combined = AudioSegment.empty()
    for audio_file in all_outputs:
        audio = AudioSegment.from_wav(audio_file)
        combined += audio

    output_combined_path = Final_OUTPUT_AUIDO
    combined.export(output_combined_path, format="wav")

    print(f"Audio generation completed: {output_combined_path}")
    return output_combined_path

@app.route('/translate', methods=['POST'])
def translate():
    """
    Handle POST requests for translation and text-to-speech conversion.

    Expected JSON payload:
    {
        "prompt": "Text to be converted to speech",
        "language": "en",
        "trans_lang": "es",
        "audio_file_pth": "path/to/audio.wav",
        "mic_file_path": "path/to/mic_audio.wav",
        "use_mic": false,
        "voice_cleanup": false,
        "no_lang_auto_detect": false
    }

    :return: Response containing the path to the generated audio file or error message.
    """
    data = request.json

    # Extract and validate parameters
    prompt = data.get('prompt')
    language = data.get('language')
    trans_lang = data.get('trans_lang')
    audio_file_pth = data.get('audio_file_pth')
    mic_file_path = data.get('mic_file_path')
    use_mic = data.get('use_mic', False)
    voice_cleanup = data.get('voice_cleanup', False)
    no_lang_auto_detect = data.get('no_lang_auto_detect', False)

    if not prompt or not language or not trans_lang:
        return jsonify({'error': 'Missing required parameters.'}), 400

    # Handle audio transcription, translation, and speech synthesis
    if not use_mic and not audio_file_pth:
        return jsonify({'error': 'Audio file path is required when not using microphone.'}), 400

    transcribed_text = None
    if not prompt:
        if not audio_file_pth and not mic_file_path:
            return jsonify({'error': 'No audio file provided.'}), 400

        if use_mic:
            transcribed_text = transcribe_audio(mic_file_path, language, no_lang_auto_detect)
        else:
            transcribed_text = transcribe_audio(audio_file_pth, language, no_lang_auto_detect)

        if not transcribed_text:
            return jsonify({'error': 'Unable to transcribe audio.'}), 400

    if not transcribed_text:
        transcribed_text = prompt

    translated_text = translate_text(transcribed_text, language, trans_lang, no_lang_auto_detect)
    if not translated_text:
        return jsonify({'error': 'Unable to translate text.'}), 400

    audio_file = predict(translated_text, trans_lang, audio_file_pth, mic_file_path, use_mic, voice_cleanup, no_lang_auto_detect)
    if not audio_file:
        return jsonify({'error': 'Unable to generate speech.'}), 500

    return send_file(audio_file, mimetype="audio/wav", as_attachment=True, attachment_filename="output.wav")

@app.route('/get_path_translated', methods=['GET'])
def get_path_translated():
    """
    Endpoint to retrieve the path of the output file from the output_translated operation.

    Returns:
        JSON response with the status and the path to the output file, if it exists.

    Raises:
        Exception: If there is an error retrieving the file path.
    """
    try:
        # Define the path to the output file (replace with the correct path and filename)
        output_path = "/tmp/translated/output_translated.wav"

        # Check if the output file exists
        if os.path.exists(output_path):
            return jsonify({
                'status': 'success',
                'message': 'Output file path retrieved successfully',
                'output_path': output_path
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Output file not found'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Define the index route
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to the Flask API!",
        "status": "success"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
