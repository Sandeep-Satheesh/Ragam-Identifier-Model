import subprocess
import sys
import argparse
import logging
import src.main as main

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("GPU is available.")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")
            logger.info(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            return True
        else:
            logger.error("GPU is not available.")
            return False
    except ImportError:
        logger.error("PyTorch is not installed. Cannot check GPU availability.")
        return False

def check_demucs():
    try:
        import demucs
        logger.info(f"Demucs version: {demucs.__version__}")
        return True
    except ImportError:
        logger.error("Demucs is not installed. Please install it to proceed.")
        return False
    
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        logger.info("FFmpeg is installed and accessible.")
        return True
    except FileNotFoundError:
        logger.info("FFmpeg is not found in the system's PATH.")
    except subprocess.CalledProcessError as e:
        logger.info(f"Error running FFmpeg: {e}")
        logger.info(f"Output: {e.stdout.decode()}")
        logger.info(f"Error Output: {e.stderr.decode()}")

    return False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to audio file or YouTube URL")
    parser.add_argument("--output", type=str, default='./raw/', help="Where separated stems go")
    args = parser.parse_args()
    
    if not args.input:
        logger.error("Input path is required. Please provide an audio file or YouTube URL.")
        sys.exit(1)
    
    logger.info("Running checks...")
    if not (check_gpu() and check_demucs() and check_ffmpeg()):
        logger.error("This script requires a GPU for processing. Exiting.")
        sys.exit(1)

    main.start_ragam_identification(args)   
