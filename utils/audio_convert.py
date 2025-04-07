from pydub import AudioSegment
import tempfile
import io

def convert_webm_to_wav(webm_bytes: bytes) -> str:
    audio_segment = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_segment.export(tmp_wav.name, format="wav")

    return tmp_wav.name
