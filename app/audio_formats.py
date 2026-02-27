import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional


OUTPUT_FORMAT_MAP = {
    "mp3_44100_128": {
        "codec": "libmp3lame",
        "bitrate": "128k",
        "sample_rate": "44100",
        "mime": "audio/mpeg",
        "ext": "mp3",
    },
    "mp3_44100_64": {
        "codec": "libmp3lame",
        "bitrate": "64k",
        "sample_rate": "44100",
        "mime": "audio/mpeg",
        "ext": "mp3",
    },
    "mp3_22050_32": {
        "codec": "libmp3lame",
        "bitrate": "32k",
        "sample_rate": "22050",
        "mime": "audio/mpeg",
        "ext": "mp3",
    },
    "wav": {
        "codec": "pcm_s16le",
        "bitrate": None,
        "sample_rate": "16000",
        "mime": "audio/wav",
        "ext": "wav",
    },
    "pcm_16000": {
        "codec": "pcm_s16le",
        "bitrate": None,
        "sample_rate": "16000",
        "mime": "audio/wav",
        "ext": "wav",
    },
    "pcm_24000": {
        "codec": "pcm_s16le",
        "bitrate": None,
        "sample_rate": "24000",
        "mime": "audio/wav",
        "ext": "wav",
    },
    "ogg_vorbis": {
        "codec": "libvorbis",
        "bitrate": "128k",
        "sample_rate": "44100",
        "mime": "audio/ogg",
        "ext": "ogg",
    },
    "flac": {
        "codec": "flac",
        "bitrate": None,
        "sample_rate": "44100",
        "mime": "audio/flac",
        "ext": "flac",
    },
}


def get_format_info(output_format: str) -> dict:
    if output_format not in OUTPUT_FORMAT_MAP:
        return OUTPUT_FORMAT_MAP["mp3_44100_128"]
    return OUTPUT_FORMAT_MAP[output_format]


def encode_audio(
    input_path: Path, output_format: str = "mp3_44100_128"
) -> Tuple[Path, str]:
    format_info = get_format_info(output_format)

    output_path = input_path.with_suffix(f".{format_info['ext']}")

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    if format_info["codec"]:
        cmd.extend(["-acodec", format_info["codec"]])

    if format_info["bitrate"]:
        cmd.extend(["-b:a", format_info["bitrate"]])

    if format_info["sample_rate"]:
        cmd.extend(["-ar", format_info["sample_rate"]])

    if output_format == "wav" or output_format.startswith("pcm_"):
        cmd.extend(["-ac", "1"])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed: {result.stderr}")

    return output_path, format_info["mime"]


def get_content_type(output_format: str) -> str:
    return get_format_info(output_format)["mime"]
