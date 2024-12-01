# Define model constants
MODEL_CHECKPOINTS = {
    # German bigvgan checkpoints
    "kisz-german-bigvgan-295000": "/home/johanna.reiml/F5TTS-German/model_295000.pt",
    "kisz-german-bigvgan-430000": "/home/johanna.reiml/F5TTS-German/model_430000.pt",
    "kisz-german-bigvgan-550000": "/home/johanna.reiml/F5TTS-German/model_550000.pt",
    "kisz-german-bigvgan-615000": "/home/johanna.reiml/F5TTS-German/model_615000.pt",
    # English checkpoints
    "original-english-vocos": "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt",
    "original-english-bigvgan": "hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt"
}

MODEL_METADATA = {
    # German bigvgan checkpoints
    "kisz-german-bigvgan-295000": {
        "vocoder_name": "bigvgan",
        "language": "de"
    },
    "kisz-german-bigvgan-430000": {
        "vocoder_name": "bigvgan",
        "language": "de"
    },
    "kisz-german-bigvgan-550000": {
        "vocoder_name": "bigvgan",
        "language": "de"
    },
    "kisz-german-bigvgan-615000": {
        "vocoder_name": "bigvgan",
        "language": "de"
    },
    # English checkpoints
    "original-english-vocos": {
        "vocoder_name": "vocos",
        "language": "en"
    },
    "original-english-bigvgan": {
        "vocoder_name": "bigvgan",
        "language": "en"
    }
}
