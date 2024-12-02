import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

import torch
from cached_path import cached_path
from f5_tts.demo.constants import MODEL_CHECKPOINTS, MODEL_METADATA
from f5_tts.demo.model import TTSModel, DeepFilterNetModel
from f5_tts.demo.preprocess import transcribe_with_cache, remove_silence
from f5_tts.infer.utils_infer import load_vocoder
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Init models
    app.state.denoise_model = DeepFilterNetModel()
    app.state.tts_models: Dict[str, TTSModel] = {}
    app.state.vocoders: Dict[str, torch.nn.Module] = {}

    # Init locks
    app.state.tts_model_locks: Dict[str, asyncio.Lock] = {
        model_id: asyncio.Lock() for model_id in MODEL_METADATA
    }
    vocoder_names = {metadata["vocoder_name"] for metadata in MODEL_METADATA.values()}
    app.state.vocoder_locks: Dict[str, asyncio.Lock] = {
        vocoder_name: asyncio.Lock() for vocoder_name in vocoder_names
    }

    yield

    # Cleanup on shutdown
    app.state.denoise_model = None
    app.state.tts_models = None
    app.state.vocoders = None
    app.state.tts_model_locks = None
    app.state.vocoder_locks = None


app = FastAPI(lifespan=lifespan)


async def get_tts_model(model_id: str) -> TTSModel:
    if model_id not in app.state.tts_models:
        model_info = MODEL_METADATA[model_id]
        vocoder = await get_vocoder(model_info["vocoder_name"])

        async with app.state.tts_model_locks[model_id]:
            if model_id not in app.state.tts_models:
                ckpt_file = str(cached_path(MODEL_CHECKPOINTS[model_id]))
                app.state.tts_models[model_id] = TTSModel(
                    ckpt_file=ckpt_file,
                    vocab_file="",
                    vocoder=vocoder,
                    vocoder_name=model_info["vocoder_name"],
                )
    return app.state.tts_models[model_id]


async def get_vocoder(vocoder_name: str) -> torch.nn.Module:
    if vocoder_name not in app.state.vocoders:
        async with app.state.vocoder_locks[vocoder_name]:
            if vocoder_name not in app.state.vocoders:
                app.state.vocoders[vocoder_name] = load_vocoder(vocoder_name=vocoder_name)
    return app.state.vocoders[vocoder_name]


@app.post("/preprocess/noise")
async def preprocess_noise(ref_audio: UploadFile = File(...)) -> StreamingResponse:
    ref_audio_bytes = await ref_audio.read()
    denoised_audio_bytes = await app.state.denoise_model.denoise(ref_audio_bytes)
    return StreamingResponse(denoised_audio_bytes, media_type="audio/wav")


@app.post("/preprocess/silence")
async def preprocess_silence(
        ref_audio: UploadFile = File(...),
        use_cuts: bool = Form(True),
        silence_threshold: int = Form(-42),
        max_reference_ms: int = Form(20000)
) -> StreamingResponse:
    ref_audio_bytes = await ref_audio.read()
    desilenced_audio_bytes = remove_silence(
        audio_bytes=ref_audio_bytes,
        use_cuts=use_cuts,
        silence_threshold=silence_threshold,
        max_reference_ms=max_reference_ms,
    )
    return StreamingResponse(desilenced_audio_bytes, media_type="audio/wav")


@app.post("/preprocess/transcribe")
async def preprocess_transcribe(ref_audio: UploadFile = File(...)) -> str:
    ref_audio_bytes = await ref_audio.read()
    ref_text = transcribe_with_cache(ref_audio_bytes)
    return ref_text


@app.get("/models")
async def get_models():
    return MODEL_METADATA


@app.post("/models/{model_id}/generate")
async def generate(
        model_id: str,
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(...),
        text: str = Form(...),
        target_rms: float = Form(0.1),
        cross_fade_duration: float = Form(0.15),
        nfe_step: int = Form(32),
        cfg_strength: float = Form(2.0),
        sway_sampling_coef: int = Form(-1),
        speed: float = Form(1.0),
) -> StreamingResponse:
    if model_id not in MODEL_METADATA:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model ID: {model_id}. Available models: {list(MODEL_METADATA.keys())}"
        )

    model = await get_tts_model(model_id)
    ref_audio_bytes = await ref_audio.read()
    gen_audio_bytes = model.sample(
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        text=text,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
    )
    return StreamingResponse(gen_audio_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
