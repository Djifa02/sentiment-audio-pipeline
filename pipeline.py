# ============================================================
# PIPELINE.PY
# Auteure : Adji Fatou NGOM
# Description : Pipeline complet Audio → Texte → Sentiment
# ============================================================

from audio import transcribe_audio
from sentiment import analyze_sentiment


def run_pipeline(file_path: str):
    """Execute le pipeline complet audio vers sentiment."""
    transcription          = transcribe_audio(file_path)
    sentiment, score, details = analyze_sentiment(transcription)
    return {
        "transcription" : transcription,
        "sentiment"     : sentiment,
        "score"         : score,
        "details"       : details
    }


if __name__ == "__main__":
    print("\nTest du pipeline complet...")
    for fichier in ["positif", "negatif", "neutre"]:
        print(f"\n--- {fichier.upper()} ---")
        resultat = run_pipeline(f"test_audio/{fichier}.mp3")
        print(f"Transcription : {resultat['transcription']}")
        print(f"Sentiment     : {resultat['sentiment']}")
        print(f"Score         : {resultat['score']}")
        print(f"Details       : {resultat['details']}")