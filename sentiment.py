# ============================================================
# SENTIMENT.PY
# Auteure : Adji Fatou NGOM
# Description : Analyse de sentiment avec BERT
# ============================================================

from transformers import pipeline

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

LABEL_MAP = {
    "positive" : "positif",
    "negative" : "negatif",
    "neutral"  : "neutre"
}

print("Chargement du modele de sentiment...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model = SENTIMENT_MODEL,
    top_k = None
)
print("Modele de sentiment charge !")


def analyze_sentiment(text: str):
    """Analyse le sentiment d un texte et retourne sentiment score et details."""
    if not text or text.strip() == "":
        raise ValueError("Le texte est vide.")
    results   = sentiment_pipeline(text)[0]
    best      = max(results, key=lambda x: x['score'])
    sentiment = LABEL_MAP.get(
        best['label'].lower(), best['label']
    )
    score   = round(best['score'], 4)
    details = {
        LABEL_MAP.get(r['label'].lower(), r['label']): round(r['score'], 4)
        for r in results
    }
    return sentiment, score, details


if __name__ == "__main__":
    textes = [
        "je suis tres satisfait de votre service merci beaucoup",
        "je suis tres decu de votre service votre equipe ne repond jamais je ne suis pas content du tout",
        "je vous appelle au sujet de ma facture du mois dernier pouvez vous me donner le montant exact"
    ]
    print("\nTest d analyse de sentiment...")
    for texte in textes:
        sentiment, score, details = analyze_sentiment(texte)
        print(f"\nTexte     : {texte}")
        print(f"Sentiment : {sentiment}")
        print(f"Score     : {score}")
        print(f"Details   : {details}")