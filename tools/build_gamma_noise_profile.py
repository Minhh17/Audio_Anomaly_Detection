# tools/build_gamma_noise_profile.py
import glob, os, argparse, numpy as np, soundfile as sf
from gammatone import gtgram

def gamma_profile(files, sr, window_time, hop_time, channels, f_min):
    acc, n = None, 0
    for wav in files:
        audio, srx = sf.read(wav)
        if srx != sr:
            import librosa
            audio = librosa.resample(audio, srx, sr)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        g = gtgram.gtgram(audio, sr, window_time, hop_time, channels, f_min) # (T,C)
        g = g.mean(0)                     # trung bình theo thời gian  → (C,)
        acc = g if acc is None else acc + g
        n += 1
    return (acc / n).astype("float32")    # (C,)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise_dir", required=True)
    ap.add_argument("--out", default="gamma_noise_profile.npy")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--window", type=float, default=0.064)
    ap.add_argument("--hop", type=float, default=0.032)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--fmin", type=float, default=50.0)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.noise_dir, "*.wav"))
    profile = gamma_profile(
        files, args.sr, args.window, args.hop, args.channels, args.fmin
    )
    np.save(args.out, profile)
    print("Saved", args.out, profile.shape)
