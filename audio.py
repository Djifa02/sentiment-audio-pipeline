# ============================================================
# AUDIO.PY
# Auteure : Adji Fatou NGOM
# Description : Pretraitement audio et transcription
#               avec Wav2Vec 2.0
# ============================================================

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_NAME   = "wav2vec2-french"
SAMPLE_RATE  = 16000
MAX_DURATION = 300

print("Chargement de Wav2Vec 2.0...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model_asr = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model_asr.eval()
print("Wav2Vec charge avec succes !")


def preprocess_audio(file_path: str):
    """Charge et prepare un fichier audio pour Wav2Vec."""
    audio, sr = librosa.load(file_path,
                             sr=SAMPLE_RATE,
                             mono=True)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def transcribe_audio(file_path: str):
    """Transcrit un fichier audio en texte avec Wav2Vec 2.0."""
    if not file_path.endswith(('.wav', '.mp3')):
        raise ValueError(
            "Format non supporte. "
            "Utilisez .wav ou .mp3"
        )
    duree = librosa.get_duration(path=file_path)
    if duree == 0:
        raise ValueError("Fichier audio vide.")
    if duree > MAX_DURATION:
        raise ValueError(
            f"Fichier trop long ({duree:.0f}s). "
            "Duree maximale : 5 minutes."
        )
    audio = preprocess_audio(file_path)
    if np.max(np.abs(audio)) < 0.01:
        raise ValueError(
            "Audio silencieux detecte. "
            "Veuillez fournir un audio avec de la parole."
        )
    inputs = processor(
        audio,
        sampling_rate  = SAMPLE_RATE,
        return_tensors = "pt"
    )
    with torch.no_grad():
        logits = model_asr(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()


if __name__ == "__main__":
    print("\nTest de transcription sur les 3 fichiers audio...")
    for fichier in ["positif", "negatif", "neutre"]:
        result = transcribe_audio(f"test_audio/{fichier}.mp3")
        print(f"{fichier} : {result}")