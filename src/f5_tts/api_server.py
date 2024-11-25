import io
import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List
from zipfile import ZipFile

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from f5_tts.model.backbones.dit import DiT

PREDEFINED_SCENARIOS = {
    'scenario1': [
        "Hey! Ich bin auf der ey eye at eytsch pih eye Konferenz und stehe hier vor einem Exponat, das meine Stimme imitieren kann. Ich merke gerade, dass sich das erschreckend echt anhört. Dabei habe ich diesen Text hier nie gesagt.",
        "Robokowls und kah ih gestützte Stimmimitation bieten kostengünstige und effiziente Kommunikationsmöglichkeiten – können jedoch bei Missbrauch zu erheblichem Schaden führen.",
        "Und genau deswegen ist es eben auch so wichtig, dass wir heute hier auf der ey eye at eytsch pih eye Konferenz über vertrauenswürdige kah ih sprechen."
    ],
    'scenario2': [
        "Hey, gut, dass ich Sie erreiche. Ich habe einen familiären Notfall und kann nicht ins Büro kommen. Können Sie mich bei einer Sache unterstützen?",
        "Ein wichtiger Kunde hat mir gerade mitgeteilt, dass noch eine Zahlung von uns aussteht. Der ist schon richtig wütend und droht mit einer Verzugsstrafe.",
        "Könnten Sie das bitte umgehend erledigen? Sie erhalten gleich eine Mail von unserem Vertragspartner, darin sind alle Informationen . Die Geschäftsleitung ist schon informiert. Ich melde mich im Laufe des Tages. Danke für Ihre Unterstützung und bis später."
    ],
    'scenario3': [
        "Hey Mama, alles okay? Du, ich habe nicht viel Zeit, von daher ganz kurz ... Dein Arzt hat mich angerufen. Du musst für die letzte Behandlung noch einen Zuschuss leisten.",
        "Das Behandlungszentrum hat noch eine offene Rechnung, die drängeln gerade etwas. Da müssen wir jetzt schnell sein, sonst gibt's Mahngebühren oben drauf. Du müsstest dreihundert vier und sechzig Euro überweisen. Du erhältst dazu gleich eine Mail, okay?",
        "Musst du dir jetzt auch nicht alles merken. In der Mail steht dann alles drin. Ist wichtig, dass du das gleich erledigst. Du machst das dann heute, okay? Ich melde mich später, weil ich bin etwas in Eile. Hab dich lieb!"
    ]
}


class ScenarioText(BaseModel):
    lines: List[str]


class BatchScenarios(BaseModel):
    scenarios: Dict[str, ScenarioText]


class TTSProcessor:
    def __init__(self, ckpt_file, vocab_file, vocoder_name="bigvgan", device=None, dtype=torch.float32, nfe_step=64):
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

        # Load the vocoder
        self.vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False)
        self.vocoder_name = vocoder_name

        # Set sampling rate
        self.sampling_rate = 24000
        
        # Set number of inference steps
        self.nfe_step = nfe_step

        # Perform warm-up
        self._warm_up()

    def _warm_up(self):
        """Warm up the model with a dummy input"""
        print("Warming up the model...")

        # Create a dummy 1-second silence WAV file
        dummy_audio = torch.zeros([1, 24000])  # 1 second of silence at 24kHz

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            torchaudio.save(temp_audio_file.name, dummy_audio, 24000)
            temp_audio_path = temp_audio_file.name
        try:
            preprocessed_ref_audio, preprocessed_ref_text = preprocess_ref_audio_text(temp_audio_path, "")
            audio, sr = torchaudio.load(preprocessed_ref_audio)
            audio = audio.to(self.device, dtype=self.dtype)
            with torch.no_grad():
                infer_batch_process(
                    (audio, sr), preprocessed_ref_text, ["Hello world"],
                    self.model, self.vocoder, nfe_step=1,
                    mel_spec_type=self.vocoder_name, device=self.device,
                )
            print("Warm-up completed.")
        finally:
            os.remove(temp_audio_path)
            if os.path.exists(preprocessed_ref_audio):
                os.remove(preprocessed_ref_audio)

    def preprocess_reference(self, ref_audio_bytes, ref_text):
        """Preprocess reference audio once"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            temp_audio_file.write(ref_audio_bytes)
            temp_audio_file.flush()
            temp_audio_file_path = temp_audio_file.name
        try:
            # If ref_text is empty, transcribe the ref_audio
            if not ref_text:
                ref_text = ""
            preprocessed_ref_audio, preprocessed_ref_text = preprocess_ref_audio_text(
                temp_audio_file_path, ref_text
            )
            audio, sr = torchaudio.load(preprocessed_ref_audio)
            return audio.to(self.device, dtype=self.dtype), sr, preprocessed_ref_text
        finally:
            os.remove(temp_audio_file_path)
            if os.path.exists(preprocessed_ref_audio):
                os.remove(preprocessed_ref_audio)

    def generate_audio_batch(self, ref_audio, sr, ref_text, texts):
        """Generate audio using preprocessed reference"""
        with torch.no_grad():
            generated_audio_waveform, final_sample_rate, _ = infer_batch_process(
                (ref_audio, sr), ref_text, texts,
                self.model, self.vocoder, nfe_step=self.nfe_step,
                mel_spec_type=self.vocoder_name, device=self.device,
            )
            if not isinstance(generated_audio_waveform, torch.Tensor):
                generated_audio_waveform = torch.from_numpy(generated_audio_waveform)
            if generated_audio_waveform.dim() == 1:
                generated_audio_waveform = generated_audio_waveform.unsqueeze(0)
        return generated_audio_waveform, final_sample_rate


async def process_batch_generation(processor, ref_audio_bytes, ref_text, scenarios_dict):
    """Common function for batch audio generation"""
    ref_audio, sr, preprocessed_ref_text = processor.preprocess_reference(ref_audio_bytes, ref_text)
    zip_buffer = io.BytesIO()

    with ZipFile(zip_buffer, 'w') as zip_file:
        for scenario_name, lines in scenarios_dict.items():
            for i, text in enumerate(lines, 1):
                try:
                    texts = [text]
                    audio_waveform, sample_rate = processor.generate_audio_batch(
                        ref_audio=ref_audio,
                        sr=sr,
                        ref_text=preprocessed_ref_text,
                        texts=texts
                    )
                    audio_bytes = io.BytesIO()
                    torchaudio.save(audio_bytes, audio_waveform, sample_rate, format='wav')
                    audio_bytes.seek(0)
                    filename = f"{scenario_name}_line{i}.wav"
                    zip_file.writestr(filename, audio_bytes.getvalue())
                    print(f"Generated audio for {filename}")
                except Exception as e:
                    print(f"Error generating {scenario_name}_line{i}: {str(e)}")
                    continue

    zip_buffer.seek(0)
    return zip_buffer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    ckpt_file = os.getenv("MODEL_FILEPATH", "/home/johanna.reiml/model_34000.pt")
    app.state.processor = TTSProcessor(ckpt_file=ckpt_file, vocab_file="", dtype=torch.float32)
    yield
    app.state.processor = None


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(""),
        text: str = Form(...)
) -> StreamingResponse:
    ref_audio_bytes = await ref_audio.read()
    texts = [text]
    ref_audio, sr, preprocessed_ref_text = app.state.processor.preprocess_reference(ref_audio_bytes, ref_text)
    audio_waveform, sample_rate = app.state.processor.generate_audio_batch(
        ref_audio=ref_audio,
        sr=sr,
        ref_text=preprocessed_ref_text,
        texts=texts
    )
    audio_bytes = io.BytesIO()
    torchaudio.save(audio_bytes, audio_waveform, sample_rate, format='wav')
    audio_bytes.seek(0)
    return StreamingResponse(audio_bytes, media_type="audio/wav")


@app.post("/generate_batch")
async def generate_batch(
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(""),
        scenarios: str = Form(...)
) -> StreamingResponse:
    scenarios_dict = json.loads(scenarios)
    zip_buffer = await process_batch_generation(
        app.state.processor, await ref_audio.read(), ref_text, scenarios_dict
    )
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=scenarios.zip"}
    )


@app.post("/generate_predefined")
async def generate_predefined(
        ref_audio: UploadFile = File(...),
        ref_text: str = Form("")
) -> StreamingResponse:
    zip_buffer = await process_batch_generation(
        app.state.processor, await ref_audio.read(), ref_text, PREDEFINED_SCENARIOS
    )
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=scenarios.zip"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
