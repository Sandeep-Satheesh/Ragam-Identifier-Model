import subprocess
import sys
import os
import yt_dlp
import logging
import librosa
import numpy as np
from urllib.parse import urlparse, parse_qs
from . import constants as const

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def download_audio_from_youtube(url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    parsed_url = urlparse(url)
    query_parameters = parse_qs(parsed_url.query)
    video_id = query_parameters.get('v', [''])[0]  # Get 'name' and default to empty string if not found

    temp_file = os.path.join(output_dir, video_id + ".wav")
    
    if os.path.exists(temp_file):
        logger.info(f"\t--> Using cached audio file: {temp_file}")
        return temp_file
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_file[0:-4],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    try:
        logger.info(f"\t--> Downloading audio from YouTube: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info(f"Downloaded audio to: {temp_file}")
    except Exception as e:
        logger.error(f"Failed to download YouTube audio: {e}")
        sys.exit(1)
    return temp_file

def load_vocals_track(track_path, duration, offset):
    logger.info(f"\t--> Loading {duration} seconds from the sample starting from offset {offset} seconds, into memory...")
    y, sr = librosa.load(track_path, sr=None, mono=True, duration=duration, offset=offset)  # Load first 'duration' seconds
    logger.info(f"\t--> Resampling at {const.SAMPLE_RATE / 1000} kHz...")
    y = y / np.max(np.abs(y))
    y = librosa.resample(y, orig_sr=sr, target_sr=const.SAMPLE_RATE)  # Resample to 16 kHz

    #Trim silence
    logger.info(f"\t--> Trimming silence lower than {const.AUDIO_TOP_DB} dB from peak...")
    y, _ = librosa.effects.trim(y, top_db=const.AUDIO_TOP_DB)  # top_db can be tuned
    intervals = librosa.effects.split(y, top_db=const.AUDIO_TOP_DB)  # adjust top_db (20–40) to tune sensitivity

    # Concatenate the kept regions
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])

    # logger.info(f"\t--> Generating Mel spectogram...")
    # mel = librosa.feature.melspectrogram(y, sr=16000, n_mels=128, hop_length=160)
    # mel = librosa.power_to_db(mel).T   # (T, n_mels)
    return non_silent_audio, const.SAMPLE_RATE
    

def preprocess(input_path, output_dir, desired_duration=const.AUDIO_LENGTH_SEC, offset=const.AUDIO_OFFSET_SEC):
    try:
        logger.info("\t--> Sourcing raw audio...")
        src_file_path = download_audio_from_youtube(input_path, output_dir)
        vocals_file_path = os.path.join(output_dir, "htdemucs", os.path.splitext(os.path.basename(src_file_path))[0], "vocals.wav")

        if not os.path.exists(vocals_file_path):
            logger.info("\t--> Separating audio into vocals...")
            subprocess.run([
                sys.executable, "-m", "demucs.separate",
                "-n", "htdemucs",
                "-o", output_dir,
                src_file_path
            ], check=True)

        logger.info("\t--> Preprocessing vocals track...")
        return load_vocals_track(vocals_file_path, desired_duration, offset)
    except Exception as e:
        logger.error(f"Demucs separation failed: {e}")
        sys.exit(1)