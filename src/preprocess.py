import subprocess
import sys
import os
import yt_dlp
import logging
import librosa
import numpy as np
from urllib.parse import urlparse, parse_qs

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

def load_vocals_track(track_path, duration=10, offset=0):
    logger.info(f"\t--> Loading {duration} seconds from the sample starting from offset {offset} seconds, into memory...")
    y, sr = librosa.load(track_path, sr=None, mono=True, duration=duration, offset=offset)  # Load first 'duration' seconds
    logger.info(f"\t--> Resampling at 16 kHz and trimming silence...")
    y = y / np.max(np.abs(y))
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)  # Resample to 16 kHz
    #Trim leading and trailing silence
    y, _ = librosa.effects.trim(y)  # top_db can be tuned

    return y, 16000
    

def preprocess(input_path, output_dir, desired_duration=10, offset=0):
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