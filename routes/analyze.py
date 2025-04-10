import io
import os

import colorsys
import librosa
import numpy as np
import soundfile as sf
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from utils.audio_convert import convert_webm_to_wav

router = APIRouter()

# 사용자 음성 기반 목소리 색상 추출
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

        analysis_fmin = 100
        analysis_fmax = 250

        pitch = np.median(librosa.yin(
            y,
            fmin=analysis_fmin,
            fmax=analysis_fmax,
            sr=sr,
            frame_length=2048,
            center=False
            ))
        
        energy = np.mean(librosa.feature.rms(y=y))
        jitter = np.std(librosa.zero_crossings(y, pad=False))

        hue = (float(pitch) - analysis_fmin) / (analysis_fmax - analysis_fmin) * 360
        saturation = max(0.2, min(1.0, float(energy) * 100))

        base_value = 0.7  # 기본 밝기 기준 올림
        jitter_penalty = min(0.4, float(jitter) * 1.5)
        value = max(0.5, base_value + (0.4 - jitter_penalty))  # 색의 밝기

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
