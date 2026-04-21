import whisper
import sip_whisper

def download_models():
    for t in whisper.available_models():
        whisper.load_model(t)
        sip_whisper.load_model(t)
