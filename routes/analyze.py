from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import colorsys
import soundfile as sf
import io
from utils.audio_convert import convert_webm_to_wav
import os

router = APIRouter()

@router.post("/upload-audio")
def upload_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = file.file.read()
        file_ext = file.filename.split('.')[-1].lower()

        if file_ext == "webm":
            tmp_wav_path = convert_webm_to_wav(audio_bytes)
            y, sr = librosa.load(tmp_wav_path, sr=22050)
            os.remove(tmp_wav_path)
        else:
            audio_io = io.BytesIO(audio_bytes)
            y, sr = sf.read(audio_io)
            if y.ndim > 1:
                y = librosa.to_mono(y.T)

        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)
        target_sr = 22050
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

        pitch = np.median(librosa.yin(y, fmin=80, fmax=300, sr=sr, frame_length=2048, win_length=1024, center=False))
        energy = np.mean(librosa.feature.rms(y=y))
        jitter = np.std(librosa.zero_crossings(y, pad=False))

        hue = (float(pitch) - 80) / (300 - 80) * 360
        saturation = max(0.2, min(1.0, float(energy) * 100))
        value = max(0.3, 1.0 - min(0.7, float(jitter) * 2))

        r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))

        return JSONResponse({
            "hex_color": hex_color,
            "features": {
                "pitch": round(float(pitch), 2),
                "energy": round(float(energy), 4),
                "jitter": round(float(jitter), 4)
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
