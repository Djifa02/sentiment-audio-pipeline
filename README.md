# Détection Automatique de Sentiment dans les Appels Vocaux

## Auteure
Adji Fatou NGOM

## Description

Ce projet développe un pipeline automatisé qui transcrit des fichiers audio en texte via Wav2Vec 2.0 puis analyse le sentiment du texte via BERT pour détecter si le client est satisfait, mécontent ou neutre.

## Architecture

```
Audio (.wav/.mp3)
        ↓
Prétraitement (16 kHz, mono, normalisation)
        ↓
Transcription Wav2Vec 2.0
        ↓
Analyse Sentiment BERT
        ↓
Classification (positif, négatif, neutre) + score de confiance
```

## Modèles utilisés

| Modèle         | Lien | Justification |
|----------------|------|---------------|
| Wav2Vec 2.0    | [jonatasgrosman/wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french) | Modèle ASR pré-entraîné sur le français. Entraîné sur 53 langues dont le français. Meilleure transcription pour nos audios en français |
| BERT Sentiment | [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | Variante de BERT fine-tuné pour le sentiment. Supporte le français. 3 classes : positif, négatif, neutre |

## Structure du projet

```
sentiment-audio-pipeline/
├── test_audio/
│   ├── positif.mp3
│   ├── negatif.mp3
│   └── neutre.mp3
├── wav2vec2-french/
├── audio.py
├── sentiment.py
├── pipeline.py
├── app.py
├── api.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/Djifa02/sentiment-audio-pipeline.git
cd sentiment-audio-pipeline
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'interface Gradio

```bash
python app.py
```

Ouvre http://127.0.0.1:7860 dans ton navigateur.

## Lancer l'API FastAPI

```bash
python api.py
```

Ouvre http://localhost:8000/docs pour tester l'API.

## Exemple d'appel API

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_audio/positif.mp3;type=audio/mpeg'
```

## Réponse JSON

```json
{
  "transcription": "je suis tres satisfait de votre service merci beaucoup",
  "sentiment": "positif",
  "score": 0.9251,
  "details": {
    "positif": 0.9251,
    "neutre": 0.0515,
    "negatif": 0.0234
  }
}
```

## Démonstration sur les fichiers de test

| Fichier | Transcription | Sentiment | Score |
|---------|--------------|-----------|-------|
| positif.mp3 | je suis tres satisfait de votre service merci beaucoup | positif | 92.5% |
| negatif.mp3 | je suis tres decu de votre service votre equipe ne repond jamais | négatif | 96.4% |
| neutre.mp3  | je vous appelle au sujet de ma facture du mois dernier | neutre | 68.0% |

## Captures d'écran

### Interface Gradio

![alt text](<Screenshot 2026-07-30 225737.png>)

### API FastAPI

![alt text](<Screenshot 2026-07-31 014543.png>)


## Limites connues

- Le modèle peut avoir du mal avec les textes très courts
- Le sentiment neutre est parfois difficile à distinguer du positif
- La qualité de la transcription dépend de la clarté de l'audio
- Les fichiers de plus de 5 minutes ne sont pas acceptés

## Cas d'usage

- Analyse automatique des appels clients
- Détection de clients mécontents en temps réel
- Rapport de satisfaction client automatisé
