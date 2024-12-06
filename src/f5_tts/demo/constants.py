# Define model constants
MODEL_CHECKPOINTS = {
    # German vocos checkpoints
    "kisz-german-vocos": "hf://aihpi/F5-TTS-German/F5TTS_Base/model_365000.safetensors",
    # German bigvgan checkpoints
    "kisz-german-bigvgan": "hf://aihpi/F5-TTS-German/F5TTS_Base_bigvgan/model_430000.safetensors",
    # English checkpoints
    "original-english-vocos": "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt",
    "original-english-bigvgan": "hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt"
}

MODEL_METADATA = {
    # German vocos checkpoints
    "kisz-german-vocos": {
        "vocoder_name": "vocos",
        "language": "de"
    },
    # German bigvgan checkpoints
    "kisz-german-bigvgan": {
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
