from . import preprocess as pre, features as feat
import logging
import sounddevice as sd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def start_ragam_identification(args):
    logger.info("1. Preprocessing input...")
    y, sr = pre.preprocess(args.input, args.output, desired_duration=45, offset=8)
    sd.play(y, samplerate=sr)

    logger.info("2. Finding the fundamental frequency (Sa frequency) for the sample...")
    f0_pitch, confidence = feat.extract_pitch(y, sr)
    sa_freq = feat.predict_sa_frequency(f0_pitch)
    logger.info(f"\t--> OUTPUT: Estimated frequency of Sa: {sa_freq} Hz")

    logger.info("3. Identifying the swaras in the song...")
    swara_counts = feat.swara_counter(f0_pitch, confidence, sa_freq)
    print(swara_counts)
    logger.info("Done.")
    sd.wait()
