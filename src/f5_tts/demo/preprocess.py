import hashlib
import io
import tempfile

import torch
from f5_tts.infer.utils_infer import remove_silence_edges, transcribe
from pydub import AudioSegment, silence

_transcribe_cache = {}


def cut_inner_silence(aseg, max_reference_ms):
    # 1. try to find long silence for clipping
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > max_reference_ms:
            print(f"Audio is over {max_reference_ms / 1000}s, clipping short. (1)")
            break
        non_silent_wave += non_silent_seg

    # 2. try to find short silence for clipping if 1. failed
    if len(non_silent_wave) > max_reference_ms:
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > max_reference_ms:
                print(f"Audio is over {max_reference_ms / 1000}s, clipping short. (2)")
                break
            non_silent_wave += non_silent_seg

    aseg = non_silent_wave
    # 3. if no proper silence found for clipping
    if len(aseg) > max_reference_ms:
        aseg = aseg[:max_reference_ms]
        print(f"Audio is over {max_reference_ms / 1000}s, clipping short. (3)")

    return aseg


def remove_silence(audio_bytes, use_cuts=True, silence_threshold=-42, max_reference_ms=20000):
    # write to file for `preprocess_ref_audio_text`
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as raw_audio_file:
        raw_audio_file.write(audio_bytes)
        raw_audio_file.flush()

        aseg = AudioSegment.from_file(raw_audio_file.name)
        if use_cuts:
            aseg = cut_inner_silence(aseg, max_reference_ms=max_reference_ms)
        aseg = remove_silence_edges(aseg, silence_threshold=silence_threshold, fadeout_duration=50)

        # Create a temporary file for processed audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as processed_audio_file:
            aseg.export(processed_audio_file.name, format="wav")

            # Read the processed audio back into a BytesIO object
            with open(processed_audio_file.name, 'rb') as f:
                processed_audio_bytes = io.BytesIO(f.read())
            processed_audio_bytes.seek(0)

            return processed_audio_bytes


@torch.no_grad()
def transcribe_with_cache(audio_bytes):
    audio_hash = hashlib.md5(audio_bytes).hexdigest()

    global _transcribe_cache
    if audio_hash not in _transcribe_cache:
        text = transcribe(audio_bytes)
        if not text.endswith(". ") and not text.endswith("。"):
            if text.endswith("."):
                text += " "
            else:
                text += ". "

        _transcribe_cache[audio_hash] = text
    return _transcribe_cache[audio_hash]

