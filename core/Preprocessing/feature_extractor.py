# -*- coding: utf-8 -*-
"""
Feature extractor that supports Gamma‑tone and Mel feature pipelines.
Now includes:
  • PCEN compression for robustness to noise
  • Optional noise‑profile subtraction (vector with same #channels)
  • Output reshaped to `target_size` (default 32 × 32) suitable for VQ‑VAE
Fixes:
  • bug where `self.impl_func` was undefined (now uses `self.feat_extr_func`)
  • correct waveform scaling
"""
from __future__ import annotations

from os import makedirs
from os.path import isdir, join, split
import json
from typing import Callable, Dict, Tuple, List

import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm
import tensorflow as tf
from gammatone import gtgram

from helper.utils import read_file_name, extract_mbe  # project helpers

# -----------------------------------------------------------------------------


class Feature_extractor:
    """Extract Gamma‑tone (or Mel) features from wav files or real‑time stream."""

    def __init__(
        self,
        src: str | None = None,
        dst: str | None = None,
        mode: str = "from_file",  # or "real_time"
        feat_type: str | None = None,  # "gamma" | "mel"
        type: str | None = None,
        # dataset / chunk params
        segment_len: float | None = None,  # seconds per chunk
        audio_len: float | None = None,  # seconds per file (optional)
        sample_per_file: int = 200,
        # Gamma params
        window_time: float | None = None,
        hop_time: float | None = None,
        channels: int | None = None,
        f_min: float | None = None,
        # Mel params
        sr: int = 16000,
        nfft: int | None = None,
        n_mel_band: int = 32,
        # profile / target shape
        noise_profile: np.ndarray | None = None,  # shape (channels,)
        target_size: Tuple[int, int] = (32, 32),
    ) -> None:
        # paths
        self.src = src
        self.dst = dst

        # usage
        self.mode = mode
        # support legacy argument name "type"
        self.feat_type = feat_type if feat_type is not None else type  # will be set/validated later

        # chunking
        self.audio_len = audio_len
        self.segment_len = segment_len  # seconds
        self.sample_per_file = sample_per_file

        # Gamma cfg
        self.window_time = window_time
        self.hop_time = hop_time
        self.channels = channels
        self.f_min = f_min

        # Mel cfg
        self.sr = sr
        self.nfft = nfft
        self.n_mel_band = n_mel_band

        # noise profile
        self.noise_profile = (
            noise_profile.astype("float32") if noise_profile is not None else None
        )
        if self.noise_profile is not None and self.noise_profile.ndim == 1:
            # ensure shape (C, 1) for broadcasting vs. time axis later
            self.noise_profile = self.noise_profile.reshape((-1, 1))

        # output size for VQ‑VAE etc.
        self.target_size = target_size

        # id dictionary (filled during extraction)
        self.data: Dict[str, List[str]] = {}

        # mapping feature type → function
        self.feat_extr_func: Dict[str, Callable[[AudioSegment], np.ndarray]] = {
            "gamma": self._get_gamma_feature,
            "mel": self._get_mel_feature,
        }

    # ------------------------------------------------------------------ PUBLIC

    def extract_feature(self, feat_type: str = "gamma") -> None:
        """Process all wav files in *src* and save features into *dst* (.npz).
        """
        self._check_directories()
        self._set_type(feat_type)
        self._extract_feature_from_file()

    def extract_feature_rt(self, audio: AudioSegment, feat_type: str = "gamma") -> np.ndarray:
        """Extract feature from an *audio* chunk in memory (real‑time)."""
        self._set_type(feat_type)
        return self.feat_extr_func[self.feat_type](audio)

    # ----------------------------------------------------------------- INTERNAL

    # ---- validation helpers --------------------------------------------------

    def _set_type(self, feat_type: str) -> None:
        if feat_type not in self.feat_extr_func:
            raise ValueError(
                f"{feat_type} is not implemented; choose one of {list(self.feat_extr_func)}"
            )
        self.feat_type = feat_type

    def _check_directories(self) -> None:
        if not self.src:
            raise ValueError("Parameter *src* (audio directory) is not provided")
        if not isdir(self.src):
            raise ValueError(f"Folder {self.src} does not exist.")
        print(f"Reading audio from {self.src}")
        makedirs(self.dst, exist_ok=True)

    # ---- extraction loops ----------------------------------------------------

    def _extract_feature_from_file(self) -> None:
        if self.segment_len is None:
            raise ValueError("segment_len (seconds) must be provided for from_file mode")

        ms_per_sample = int(self.segment_len * 1000)  # pydub uses milliseconds
        file_list = read_file_name(self.src)
        if not file_list:
            print("No wav files found – nothing to do.")
            return

        # compute expected #segments per wav (if audio_len known)
        rate = int(self.audio_len / self.segment_len) if self.audio_len else None

        file_count = len(file_list)
        num_total_segments = file_count * rate if rate else None
        num_chunks_per_npz = self.sample_per_file

        chunk_idx: List[str] = []
        feature_buf: List[np.ndarray] = []
        npz_counter, segment_counter = 0, 0

        for wav_path in tqdm(file_list, desc=f"Extracting {self.feat_type} features"):
            audio = AudioSegment.from_file(wav_path, format="wav")
            for seg_idx, chunk in enumerate(make_chunks(audio, ms_per_sample)):
                feat = self.feat_extr_func[self.feat_type](chunk)
                feature_buf.append(feat)
                fname = wav_path.split("/")[-1]
                chunk_idx.append(f"{fname[:-4]}_{seg_idx}")
                segment_counter += 1

                # save when buffer full or last iteration
                save_now = len(feature_buf) == num_chunks_per_npz
                last_iter = (num_total_segments is None) and (seg_idx == len(file_list) - 1)
                if save_now or last_iter:
                    np.savez_compressed(join(self.dst, str(npz_counter)), np.array(feature_buf))
                    feature_buf.clear()
                    npz_counter += 1

        category = f"{self.dst.split('/')[-2]}_{self.dst.split('/')[-1]}"
        self.data[category] = chunk_idx
        print(f"Saved {category} – {segment_counter} segments across {npz_counter} part files")

    # ---- feature implementations -------------------------------------------

    @staticmethod
    def _waveform_from_audiosegment(seg: AudioSegment) -> Tuple[np.ndarray, int]:
        """Convert *seg* → mono float32 ndarray in [‑1,1] and return (wave, sr)."""
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.sample_width == 2:
            samples /= 32768.0  # 16‑bit PCM
        else:  # generic scale
            samples /= float(2 ** (8 * seg.sample_width - 1))
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels)).mean(axis=1)
        return samples, seg.frame_rate

    @staticmethod
    def _pcen_numpy(
        S: np.ndarray,
        sr: int,
        hop_length: int,
        time_constant: float = 0.06,
        gain: float = 0.98,
        bias: float = 2.0,
        power: float = 0.5,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Return PCEN of ``S`` computed via simple IIR smoothing.

        Parameters ``S`` is a spectrogram with shape ``(T, C)`` where ``T`` is
        the time dimension. ``sr`` and ``hop_length`` control the smoothing
        coefficient. Implementation follows ``librosa.pcen``.
        """
        b = 1 - np.exp(-hop_length / (sr * time_constant))
        M = np.zeros_like(S, dtype=np.float32)
        M[0] = S[0]
        for t in range(1, S.shape[0]):
            M[t] = (1 - b) * M[t - 1] + b * S[t]
        return ((S / (eps + M) ** gain + bias) ** power - bias ** power)

    def _get_gamma_feature(self, seg: AudioSegment) -> np.ndarray:
        """Return denoised log‑PCEN Gamma spectrogram resized to ``target_size``."""
        wave, sr = self._waveform_from_audiosegment(seg)
        gtg = gtgram.gtgram(
            wave,
            sr,
            self.window_time,
            self.hop_time,
            self.channels,
            self.f_min,
        )  # shape (T, C)
        hop_length = int(self.hop_time * sr)
        pcen = self._pcen_numpy(
            gtg,
            sr=sr,
            hop_length=hop_length,
            time_constant=0.06,
            gain=0.98,
            bias=2.0,
            power=0.5,
        ).T  # (C, T)

        # subtract noise profile if provided
        if self.noise_profile is not None:
            pcen = np.clip(pcen - self.noise_profile, 0.0, None)

        log_pcen = np.log(pcen + 1e-6)  # (C, T)
        C, T = log_pcen.shape

        target_C, target_T = self.target_size

        # --- fix time dimension to target_T ---
        if T < target_T:
            pad = np.zeros((C, target_T - T), dtype=log_pcen.dtype)
            log_pcen = np.concatenate([log_pcen, pad], axis=1)
        else:
            log_pcen = log_pcen[:, :target_T]

        # --- resize channel dimension to target_C via linear interpolation ---
        x_old = np.linspace(0, C - 1, C)
        x_new = np.linspace(0, C - 1, target_C)
        resized = np.vstack(
            [np.interp(x_new, x_old, log_pcen[:, t]) for t in range(target_T)]
        ).T  # (target_C, target_T)

        return resized.astype(np.float32)

    def _get_mel_feature(self, seg: AudioSegment) -> np.ndarray:
        """Original Mel pipeline retained for backward compatibility."""
        wave, _ = self._waveform_from_audiosegment(seg)
        return extract_mbe(wave, self.sr, self.nfft, self.n_mel_band)

    # ---- utils --------------------------------------------------------------

    def save_id(self) -> None:
        """Save accumulated *self.data* idx map next to dst folder."""
        dst_parts = split(self.dst)
        part_name = self.dst.split("/")[-1]
        idx_path = join(dst_parts[0], f"{part_name}_idx.json")
        with open(idx_path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=2)
        print(f"Saved index json → {idx_path}")
        self.data.clear()
