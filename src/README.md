
#Project Structure for Raga Identifier

ragam_identifier/
│
├── data/                 # For storing downloaded audio samples
├── models/               # Trained model files
├── scripts/              # One-off scripts for data prep, training, testing
├── ragam_identifier/     # Main source code
│   ├── __init__.py
│   ├── preprocess.py     # Audio loading, spleeter isolation, feature extraction
│   ├── features.py       # MFCC, pitch, tonic extraction
│   ├── model.py          # DL model definition + loading
│   ├── predict.py        # Inference pipeline
│   └── utils.py          # Helper functions
├── tests/                # Unit tests
├── requirements.txt
├── .gitignore
└── main.py               # Entry point script