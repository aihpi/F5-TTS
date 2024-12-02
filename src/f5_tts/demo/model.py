import asyncio
import io

import torch
import torchaudio
from df import init_df, enhance

from f5_tts.infer.utils_infer import infer_batch_process, load_vocoder, load_model
from f5_tts.model.backbones.dit import DiT
from torchaudio.transforms import Resample


class TTSModel:
    def __init__(
            self,
            ckpt_file,
            vocab_file,
            vocoder=None,
            vocoder_name="vocos",  # "vocos" or "bigvgan"
            device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file,
            mel_spec_type=vocoder_name,
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        )

        # Load the vocoder
        if vocoder is None:
            self.vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False)
            self.vocoder_name = vocoder_name
        else:
            self.vocoder = vocoder
            self.vocoder_name = vocoder_name

    @torch.no_grad()
    def sample(
            self,
            ref_audio_bytes,
            ref_text,
            text,
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=32,
            cfg_strength=2.0,
            sway_sampling_coef=-1,
            speed=1.0,
    ):
        ref_audio = torchaudio.load(io.BytesIO(ref_audio_bytes))

        gen_audio, final_sr, _ = infer_batch_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text_batches=[text],
            model_obj=self.model,
            vocoder=self.vocoder,
            mel_spec_type=self.vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            device=self.device,
        )
        if not isinstance(gen_audio, torch.Tensor):
            gen_audio = torch.from_numpy(gen_audio)
        if gen_audio.dim() == 1:
            gen_audio = gen_audio.unsqueeze(0)

        gen_audio_bytes = io.BytesIO()
        torchaudio.save(gen_audio_bytes, gen_audio, final_sr, format='wav')
        gen_audio_bytes.seek(0)
        return gen_audio_bytes


class DeepFilterNetModel:
    def __init__(self):
        self.model, self.df_state, _ = init_df()
        self.lock = asyncio.Lock()  # Create async lock for thread safety

    @torch.no_grad()
    async def denoise(self, audio_bytes):
        target_sample_rate = 48000

        # Load and resample if the audio is not 48kHz
        audio_stream = io.BytesIO(audio_bytes)
        audio, sr = torchaudio.load(audio_stream)
        if sr != target_sample_rate:
            resample = Resample(orig_freq=sr, new_freq=target_sample_rate)
            audio = resample(audio)

        # Clean the audio with thread safety
        async with self.lock:  # Acquire lock before modifying df_state
            denoised_audio = enhance(self.model, self.df_state, audio)

        # Save the clean audio to bytes
        denoised_audio_bytes = io.BytesIO()
        torchaudio.save(denoised_audio_bytes, denoised_audio, target_sample_rate, format='wav')
        denoised_audio_bytes.seek(0)
        return denoised_audio_bytes
