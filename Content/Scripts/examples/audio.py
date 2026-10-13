"""Audio example: pitch and loudness of what a Python Audio Stream hears (microphone, a submix, or pushed
samples), computed on the GPU. A template for audio models: ut.Audio resamples each chunk to 16 kHz mono,
batched across streams; swap analyze() for a torchaudio model (e.g. torchaudio.pipelines) to classify or
transcribe.

    "audio_features"   analyze(audio) -> pitch (Hz), loudness (dBFS), text ("440 Hz, -9 dB")
"""
import torch
import unreal_torch as ut


@ut.model("audio_features")
class AudioFeatures(ut.Model):
    RATE = 16000

    @ut.export(batch=True)
    def analyze(self, audio: ut.Audio(rate=RATE)) -> dict:
        """audio: (N, samples) float32 at 16 kHz."""
        n = audio.shape[-1]
        window = torch.hann_window(n, device=audio.device)
        spectrum = torch.fft.rfft(audio * window, n=4 * n).abs()  # zero-padded: finer bins
        spectrum[:, :4] = 0  # no DC
        peak = spectrum.argmax(-1)
        # Parabolic interpolation around the peak bin
        i = peak.clamp(1, spectrum.shape[-1] - 2)
        a, b, c = (spectrum.gather(-1, (i + d).unsqueeze(-1)).squeeze(-1) for d in (-1, 0, 1))
        shift = 0.5 * (a - c) / (a - 2 * b + c).where((a - 2 * b + c).abs() > 1e-12, torch.ones_like(b))
        pitch = (i.float() + shift) * self.RATE / (4 * n)
        rms = audio.pow(2).mean(-1).sqrt()
        loudness = 20 * torch.log10(rms.clamp_min(1e-6))
        text = [f"{p:.0f} Hz, {l:.0f} dB" if l > -60 else "silence" for p, l in zip(pitch.tolist(), loudness.tolist())]
        return {"pitch": pitch, "loudness": loudness, "text": text}
