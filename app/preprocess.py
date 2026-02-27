import os
import subprocess
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Tuple
import httpx


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = "pcm_s16le"


async def download_file(url: str, dest: Path) -> Path:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


def convert_to_anchor(
    input_path: Path, output_path: Path, enable_vad: bool = True
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-ac",
            str(TARGET_CHANNELS),
            "-acodec",
            "pcm_s16le",
            "-threads",
            "4",
            str(tmp_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

        if enable_vad:
            try:
                import webrtcvad

                tmp_path = trim_vad(tmp_path, tmp_path)
            except Exception:
                pass

        tmp_path.rename(output_path)
        return output_path

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def trim_vad(input_path: Path, output_path: Path, frame_duration_ms: int = 30) -> Path:
    try:
        import webrtcvad
    except ImportError:
        return input_path

    import wave
    import numpy as np

    vad = webrtcvad.Vad(mode=3)

    with wave.open(str(input_path), "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_rate not in [8000, 16000, 32000, 48000]:
        return input_path

    frame_size = (
        int(sample_rate * frame_duration_ms / 1000) * sample_width * num_channels
    )

    voiced_frames = []
    for i in range(0, len(frames), frame_size):
        frame = frames[i : i + frame_size]
        if len(frame) < frame_size:
            break

        try:
            is_speech = vad.is_speech(frame, sample_rate)
            if is_speech:
                voiced_frames.append(frame)
        except Exception:
            continue

    if not voiced_frames:
        return input_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wf_out:
        wf_out.setnchannels(num_channels)
        wf_out.setsampwidth(sample_width)
        wf_out.setframerate(sample_rate)
        wf_out.writeframes(b"".join(voiced_frames))

    return output_path


async def preprocess_audio(
    source: Path, enable_vad: bool = True, temp_dir: Optional[Path] = None
) -> Path:
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())

    output_path = temp_dir / f"anchor_{os.urandom(8).hex()}.wav"

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, convert_to_anchor, source, output_path, enable_vad
    )


def get_audio_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip())
    return 0.0
