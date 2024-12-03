# Define model constants
MODEL_CHECKPOINTS = {
    # German vocos checkpoints
    "kisz-german-vocos-295000": "hf://aihpi/F5-TTS-German/F5TTS_Base/model_295000.safetensors",
    "kisz-german-vocos-365000": "hf://aihpi/F5-TTS-German/F5TTS_Base/model_365000.safetensors",
    "kisz-german-vocos-420000": "hf://aihpi/F5-TTS-German/F5TTS_Base/model_420000.safetensors",
    # German bigvgan checkpoints
    "kisz-german-bigvgan-295000": "hf://aihpi/F5-TTS-German/F5TTS_Base_bigvgan/model_295000.safetensors",
    "kisz-german-bigvgan-430000": "hf://aihpi/F5-TTS-German/F5TTS_Base_bigvgan/model_430000.safetensors",
    "kisz-german-bigvgan-550000": "hf://aihpi/F5-TTS-German/F5TTS_Base_bigvgan/model_550000.safetensors",
    "kisz-german-bigvgan-615000": "hf://aihpi/F5-TTS-German/F5TTS_Base_bigvgan/model_615000.safetensors",
    # English checkpoints
    "original-english-vocos": "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt",
    "original-english-bigvgan": "hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt"
}

MODEL_METADATA = {
    # German vocos checkpoints
    "kisz-german-vocos-295000": {
        "vocoder_name": "vocos",
        "language": "de"
    },
    "kisz-german-vocos-365000": {
        "vocoder_name": "vocos",
        "language": "de"
    },
    "kisz-german-vocos-420000": {
        "vocoder_name": "vocos",
        "language": "de"
    },
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
