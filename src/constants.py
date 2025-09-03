#Training configs

POOLING_METHOD = 'attention' #'attention', 'average', 'max'
TRAIN_EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 128
TRAIN_SNIPPET_SECONDS = 45
TRAIN_SNIPPET_OVERLAP_FACTOR = 0.5
TENSORBOARD_LOG_DIR = "runs/tensorboard_logs"
CHECKPOINT_DIR_PATH = "runs/checkpoints"
RESUME_TRAINING = True
EARLY_STOPPING_PATIENCE = 30
CHECKPOINT_FILE_NAME = "ragam_crnn.pth"
TRAIN_SNIPPET_DOWNSAMPLE_FACTOR = 2
TRAINING_LOG_FILE_PATH = "runs/training_logs.txt"
CHECKPOINT_INTERVAL = 10
# Augmentation config
AUGMENT_PROBS = {
    "tonic_shift": 0.5,   # 50% chance
    "tempo": 0.3,         # 30% chance
    "noise": 0.3          # 30% chance
}
AUGMENT_RANGES = {
    "tonic_shift": 250,
    "tempo": 0.25,
    "noise": 10 #cents
}

#Inference configs

HOP_LENGTH = 128  # 8 ms hop length @ 16kHz
FMIN_HZ = 70
FMAX_HZ = 1200
AUDIO_LATENCY_SEC = 0.625   # adjust by ear
SAMPLE_RATE = 16000
AUDIO_LENGTH_SEC = 60  # seconds
AUDIO_OFFSET_SEC = 0  # seconds
AUDIO_TOP_DB = 20  # for trimming silence
AUDIO_BATCH_SIZE = 2048
TORCH_MEDFILT_KERNEL_SIZE = 5
HISTOGRAM_BINS_SA_ESTIMATION = 120
AUDIO_BLOCK_SIZE = 1024  # for audio playback in visualizer
SWARA_MAPPING_TOLERANCE_CENTS = 50.0
SWARA_MAPPING_CONFIDENCE_THRESHOLD = 0.7
EXPECTED_SA_RANGE_HZ = (130.81, 261.63)  # C3 to C4 (C4=middle C)
SWARA_MAPPING_SEARCH_OCTAVES = (-2, -1, 0, 1, 2)
SWARA_MAPPING_TOLERANCES = {
    "Sa": 50.0,
    "Pa": 50.0,
    "Ri1": 30.0, "Ri2": 30.0,
    "Ga1": 30.0, "Ga2": 30.0,
    "Ma1": 35.0, "Ma2": 35.0,
    "Da1": 30.0, "Da2": 30.0,
    "Ni1": 30.0, "Ni2": 30.0,
}
SWARA_MAPPING_DEFAULT_TOLERANCE_CENTS = 40.0