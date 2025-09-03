from . import preprocess as pre, features as feat
import logging
from collections import Counter
from . import visualizer
from src.crnn import dataset, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def start_training(args):
    logger.info("1. Preparing dataset...")
    ds = dataset.CarnaticPitchDataset(args.dataset_path, snippet_seconds=args.snippet_seconds, overlap=args.snippet_overlap_factor, downsample_factor=args.snippet_downsample_factor)
    logger.info(f"\t--> OUTPUT: Total snippets: {len(ds)}")
    x, y = ds[0]
    logger.info(f"\t--> OUTPUT: Snippet shape: {x.shape}, Label id: {y}")

    logger.info("2. Starting training...")
    trainer = train.RagamTrainer(dataset=ds, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, device='cuda',pooling=args.pooling_method)
    trainer.train()

def start_ragam_identification(args):
    logger.info("1. Preprocessing input...")
    y, sr = pre.preprocess(args.input, args.output, desired_duration=90, offset=0)

    logger.info("2. Finding the fundamental frequency (Sa frequency) for the sample...")
    f0_pitch, confidence = feat.extract_pitch(y, sr)
    sa_freq = feat.predict_sa_frequency(f0_pitch, conf=confidence)
    logger.info(f"\t--> OUTPUT: Estimated frequency of Sa: {sa_freq} Hz")

    logger.info("3. Identifying the swaras in the song...")
    swaras, swara_map = feat.get_swaras_for_frames(f0_pitch, confidence, sa_freq)

    logger.info("4. Visualizing detected notes...")
    visualizer.visualize_notes(y, f0_pitch, swaras, swara_map, play_sine_audio=True)
    print(Counter(swaras))
    logger.info("Done.")
