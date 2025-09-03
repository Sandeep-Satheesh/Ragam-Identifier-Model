from . import preprocess as pre, features as feat
import logging
from collections import Counter
from . import visualizer
from src.crnn import dataset, train, inference
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def start_training(args):
    logger.info("1. Preparing dataset...")
    ds = dataset.CarnaticPitchDataset(args.dataset_path, snippet_seconds=args.snippet_seconds, overlap=args.snippet_overlap_factor, downsample_factor=args.snippet_downsample_factor, force_rebuild=False, augment=True)
    logger.info(f"\t--> OUTPUT: Total snippets: {len(ds)}")
    x, y = ds[0]
    logger.info(f"\t--> OUTPUT: Snippet shape: {x.shape}, Label id: {y}")

    logger.info("2. Starting training...")
    trainer = train.RagamTrainer(dataset=ds, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, device='cuda',pooling=args.pooling_method, )
    trainer.train()

def start_ragam_identification(args):
    logger.info("1. Preprocessing input...")
    y, sr = pre.preprocess(args.input, args.output, desired_duration=45, offset=0)

    logger.info("2. Finding the fundamental frequency (Sa frequency) for the sample...")
    f0_pitch, confidence = feat.extract_pitch(y, sr)
    sa_freq = feat.predict_sa_frequency(f0_pitch, conf=confidence)
    logger.info(f"\t--> OUTPUT: Estimated frequency of Sa: {sa_freq} Hz")

    logger.info("3. Identifying the swaras in the song...")
    #swaras, swara_map = feat.get_swaras_for_frames(f0_pitch, confidence, sa_freq)

    logger.info("4. Visualizing detected notes...")
    #visualizer.visualize_notes(y, f0_pitch, swaras, swara_map, play_sine_audio=True)

    # Build dataset (for mappings only)
    ds = dataset.CarnaticPitchDataset(args.dataset_path, snippet_seconds=45, overlap=0.5, downsample_factor=2)
    idx2raga = {i: r for r, i in ds.raga2idx.items()}

    # Load model
    model = inference.load_model('runs/checkpoints/best_model.pth', num_classes=len(idx2raga), device='cuda')
    f0_pitch = inference.preprocess_sample(f0_pitch, sa_freq, ds.snippet_frames)

    # Predict
    raga, probs, attn = inference.predict(model, f0_pitch, idx2raga, 'cuda', return_attention=True)
    print(f"Predicted Raga: {raga}")
    print(f"Top-5: {[idx2raga[i] for i in np.argsort(probs)[::-1][:5]]}")

    #print(Counter(swaras))
    logger.info("Done.")
