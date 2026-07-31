# ============================================================
# EVALUATE.PY
# Auteure : Adji Fatou NGOM
# Description : Evaluation quantitative du pipeline
#               WER pour l ASR
#               Accuracy et F1 pour le sentiment
# ============================================================

from pipeline import run_pipeline
from jiwer import wer
from sklearn.metrics import accuracy_score, f1_score, classification_report

reference_texts = {
    "test_audio/positif.mp3" : "je suis tres satisfait de votre service merci beaucoup",
    "test_audio/negatif.mp3" : "je suis tres decu de votre service votre equipe ne repond jamais je ne suis pas content du tout",
    "test_audio/neutre.mp3"  : "je vous appelle au sujet de ma facture du mois dernier pouvez vous me donner le montant exact sil vous plait"
}

reference_sentiments = {
    "test_audio/positif.mp3" : "positif",
    "test_audio/negatif.mp3" : "negatif",
    "test_audio/neutre.mp3"  : "neutre"
}


def evaluate_pipeline():
    """Evalue le pipeline sur les 3 fichiers audio de test."""
    print("\n" + "="*50)
    print("EVALUATION DU PIPELINE")
    print("="*50)

    wer_scores            = []
    true_sentiments       = []
    predicted_sentiments  = []

    for audio_path in reference_texts.keys():
        print(f"\n--- {audio_path} ---")
        resultat            = run_pipeline(audio_path)
        transcription       = resultat['transcription']
        sentiment           = resultat['sentiment']
        reference_text      = reference_texts[audio_path]
        reference_sentiment = reference_sentiments[audio_path]
        score_wer           = wer(reference_text, transcription)
        wer_scores.append(score_wer)
        true_sentiments.append(reference_sentiment)
        predicted_sentiments.append(sentiment)
        print(f"Reference     : {reference_text}")
        print(f"Transcription : {transcription}")
        print(f"WER           : {score_wer * 100:.1f}%")
        print(f"Sentiment reel    : {reference_sentiment}")
        print(f"Sentiment predit  : {sentiment}")

    wer_moyen = sum(wer_scores) / len(wer_scores)
    accuracy  = accuracy_score(true_sentiments, predicted_sentiments)
    f1        = f1_score(
        true_sentiments,
        predicted_sentiments,
        average='weighted'
    )

    print("\n" + "="*50)
    print("RESULTATS GLOBAUX")
    print("="*50)
    print(f"WER moyen  : {wer_moyen * 100:.1f}%")
    print(f"Accuracy   : {accuracy * 100:.1f}%")
    print(f"F1 Score   : {f1:.4f}")
    print("\n--- RAPPORT DE CLASSIFICATION ---")
    print(classification_report(true_sentiments, predicted_sentiments))

    return {
        "wer_moyen" : wer_moyen,
        "accuracy"  : accuracy,
        "f1"        : f1
    }


if __name__ == "__main__":
    evaluate_pipeline()