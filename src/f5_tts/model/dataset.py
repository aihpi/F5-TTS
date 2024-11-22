import json
import random
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
        }


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    split: str = "train",
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    split           - "train", "validation", or "test" for split-specific dataset loading
    """

    print(f"Loading dataset for split: {split} ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                dataset = load_from_disk(f"{rel_data_path}/{split}")
            except:  # noqa: E722
                dataset = Dataset_.from_file(f"{rel_data_path}/{split}.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            dataset = Dataset_.from_file(f"{rel_data_path}/{split}.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/{split}_duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        dataset = CustomDataset(
            dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            dataset = load_from_disk(f"{dataset_name}/{split}")
        except:  # noqa: E722
            dataset = Dataset_.from_file(f"{dataset_name}/{split}.arrow")

        with open(f"{dataset_name}/{split}_duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        dataset = CustomDataset(
            dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"{split}.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return dataset


def get_collate_fn(tokenizer, vocab_char_map=None):
    def collate_fn(batch):
        # Step 0: Mel Specs
        mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
        mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
        max_mel_length = mel_lengths.amax()

        padded_mel_specs = []
        for spec in mel_specs:  # TODO. maybe records mask for attention here
            padding = (0, max_mel_length - spec.size(-1))
            padded_spec = F.pad(spec, padding, value=0)
            padded_mel_specs.append(padded_spec)

        mel_specs = torch.stack(padded_mel_specs)

        # Step 1: Preprocess text (stored as list of chars)
        texts = ["".join(item["text"]) for item in batch]

        # Step 2: Tokenize text
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        token_ids = tokenized["input_ids"]  # [batch, token_seq_len]
        token_ids_mask = tokenized["attention_mask"]  # [batch, token_seq_len]
        offset_mapping = tokenized["offset_mapping"]  # [batch, token_seq_len, 2]

        # Step 3: Get max_token_char_len from offset_mapping (including spaces)
        max_token_char_len = 0
        for offsets in offset_mapping:
            previous_end = 0
            for start, end in offsets:
                spaces = max(0, start - previous_end)
                token_char_len = spaces + (end - start)
                max_token_char_len = max(max_token_char_len, token_char_len)
                previous_end = end

        # Step 4: Prepare character-level encoding
        batch_size, token_seq_len = token_ids.shape
        char_ids = torch.zeros(
            (batch_size, token_seq_len, max_token_char_len), dtype=torch.long
        )
        char_ids_mask = torch.zeros(
            (batch_size, token_seq_len, max_token_char_len), dtype=torch.long
        )

        for i, (text, offsets) in enumerate(zip(texts, offset_mapping)):
            previous_end = 0  # Track the end of the previous token

            for token_idx, (start, end) in enumerate(offsets):
                num_spaces = start - previous_end
                previous_end = end
                if num_spaces == 0 and start == end:  # Padding tokens
                    continue

                # Extract characters for the token
                raw_chars = text[start:end]

                # Convert characters to indices
                chars = ([vocab_char_map.get(" ") if vocab_char_map else ord(" ") for _ in range(num_spaces)] +
                          [vocab_char_map.get(c, 0) if vocab_char_map else ord(c) for c in raw_chars])
                padding = [-1] * (max_token_char_len - len(chars))
                combined = chars + padding

                assert max_token_char_len >= len(combined)

                # Assign to char_ids
                char_ids[i, token_idx, :] = torch.tensor(combined)

                # Mask: Set 1 for valid character indices, 0 for padding
                char_ids_mask[i, token_idx, :len(chars)] = 1

        return {
            "mel": mel_specs,
            "mel_lengths": mel_lengths,
            "token_ids": token_ids,
            "token_ids_mask": token_ids_mask,
            "char_ids": char_ids,
            "char_ids_mask": char_ids_mask,
        }

    return collate_fn
