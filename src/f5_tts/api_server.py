from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
import torchaudio
from transformers import AutoTokenizer

from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from f5_tts.model.backbones.dit import DiT

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse

from cached_path import cached_path
import io
import os
import tempfile

class TTSProcessor:
    def __init__(self, ckpt_file, vocab_file, vocoder_name="bigvgan", device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Load the model
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file,
            mel_spec_type=vocoder_name,  # or "bigvgan" depending on vocoder
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=self.dtype)
        token_embedding_model_name = "meta-llama/Llama-3.2-3B"
        self.tokenizer = AutoTokenizer.from_pretrained(token_embedding_model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load the vocoder
        self.vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False)
        self.vocoder_name = vocoder_name

        # Set sampling rate
        self.sampling_rate = 24000

        # Perform warm-up
        self._warm_up()

    def _warm_up(self):
        """Warm up the model with a dummy input to ensure it's ready for real-time processing."""
        print("Warming up the model...")

        # Create a dummy 1-second silence WAV file
        dummy_audio = torch.zeros([1, 24000])  # 1 second of silence at 24kHz

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            torchaudio.save(temp_audio_file.name, dummy_audio, 24000)
            temp_audio_path = temp_audio_file.name

        try:
            preprocessed_ref_audio, preprocessed_ref_text = preprocess_ref_audio_text(
                temp_audio_path,
                ""
            )

            audio, sr = torchaudio.load(preprocessed_ref_audio)
            audio = audio.to(self.device, dtype=self.dtype)

            with torch.no_grad():
                infer_batch_process(
                    (audio, sr),
                    preprocessed_ref_text,
                    ["Hello world"],  # generation text
                    self.model,
                    self.vocoder,
                    tokenizer=self.tokenizer,
                    mel_spec_type=self.vocoder_name,
                    device=self.device,
                )
            print("Warm-up completed.")

        finally:
            # Clean up temporary files
            os.remove(temp_audio_path)
            if os.path.exists(preprocessed_ref_audio):
                os.remove(preprocessed_ref_audio)

    def generate_audio(self, ref_audio_bytes, ref_text, text):
        # Write the uploaded audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            temp_audio_file.write(ref_audio_bytes)
            temp_audio_file.flush()
            temp_audio_file_path = temp_audio_file.name

        try:
            # If ref_text is empty, transcribe the ref_audio
            if not ref_text:
                ref_text = ""

            # Preprocess reference audio and text using the file path
            preprocessed_ref_audio, preprocessed_ref_text = preprocess_ref_audio_text(
                temp_audio_file_path,
                ref_text
            )

            # Load the preprocessed reference audio
            audio, sr = torchaudio.load(preprocessed_ref_audio)
            audio = audio.to(self.device, dtype=self.dtype)

            # Run inference
            with torch.no_grad():
                generated_audio_waveform, final_sample_rate, _ = infer_batch_process(
                    (audio, sr),
                    preprocessed_ref_text,
                    [text],
                    self.model,
                    self.vocoder,
                    tokenizer=self.tokenizer,
                    mel_spec_type=self.vocoder_name,
                    device=self.device,
                )
                # Convert numpy array to tensor if necessary
                if not isinstance(generated_audio_waveform, torch.Tensor):
                    generated_audio_waveform = torch.from_numpy(generated_audio_waveform)

                # Reshape to [1, samples] for mono audio
                if generated_audio_waveform.dim() == 1:
                    generated_audio_waveform = generated_audio_waveform.unsqueeze(0)

            return generated_audio_waveform, final_sample_rate

        finally:
            # Clean up the temporary file
            os.remove(temp_audio_file_path)

# fastapi
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Fetch ckpt_file from an environment variable or use a default path
    ckpt_file = os.getenv("MODEL_FILEPATH", "/mnt/raid/johanna.reiml/ckpts/F5TTS_DE_bigvgan-Fusion_Llama-3B/model_49000.pt")

    # Initialize the TTSProcessor with the provided or default checkpoint file
    app.state.processor = TTSProcessor(
        ckpt_file=ckpt_file,
        vocab_file="",
        dtype=torch.float32,
    )
    yield
    # Cleanup (if needed)
    app.state.processor = None

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    text: str = Form(...)
) -> StreamingResponse:
    ref_audio_bytes = await ref_audio.read()

    # Generate the audio
    audio_waveform, sample_rate = app.state.processor.generate_audio(ref_audio_bytes, ref_text, text)

    # Convert the generated waveform to bytes
    audio_bytes = io.BytesIO()
    torchaudio.save(audio_bytes, audio_waveform, sample_rate, format='wav')
    audio_bytes.seek(0)

    return StreamingResponse(audio_bytes, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
