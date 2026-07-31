# ============================================================
# APP.PY
# Auteure : Adji Fatou NGOM
# Description : Interface Gradio pour le pipeline
#               Audio → Transcription → Sentiment
# ============================================================

import gradio as gr
from pipeline import run_pipeline


def predict(audio_file):
    """Transcrit un fichier audio et analyse son sentiment."""
    if audio_file is None:
        return "Aucun fichier audio fourni.", "", "", ""
    try:
        resultat    = run_pipeline(audio_file)
        score_pct   = f"{resultat['score'] * 100:.1f}%"
        details_str = "\n".join([
            f"{classe} : {score * 100:.1f}%"
            for classe, score in resultat['details'].items()
        ])
        return (
            resultat['transcription'],
            resultat['sentiment'],
            score_pct,
            details_str
        )
    except ValueError as e:
        return str(e), "", "", ""
    except Exception as e:
        return f"Erreur inattendue : {str(e)}", "", "", ""


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Detection Automatique de Sentiment dans les Appels Vocaux
    ### Pipeline Audio vers Sentiment avec Wav2Vec 2.0 et BERT
    Uploadez un fichier audio (.wav ou .mp3) de maximum 5 minutes.
    Le systeme va transcrire l audio et analyser le sentiment.
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label = "Fichier audio",
                type  = "filepath"
            )
            submit_btn = gr.Button(
                "Analyser",
                variant = "primary"
            )

        with gr.Column():
            transcription_output = gr.Textbox(
                label = "Transcription"
            )
            sentiment_output = gr.Textbox(
                label = "Sentiment detecte"
            )
            score_output = gr.Textbox(
                label = "Score de confiance"
            )
            details_output = gr.Textbox(
                label = "Probabilites par classe",
                lines = 3
            )

    gr.Examples(
        examples = [
            ["test_audio/positif.mp3"],
            ["test_audio/negatif.mp3"],
            ["test_audio/neutre.mp3"]
        ],
        inputs = audio_input
    )

    gr.Markdown("""
    ---
    Auteure : Adji Fatou NGOM
    Modele ASR : jonatasgrosman/wav2vec2-large-xlsr-53-french
    Modele Sentiment : cardiffnlp/twitter-xlm-roberta-base-sentiment
    """)

    submit_btn.click(
        fn      = predict,
        inputs  = audio_input,
        outputs = [
            transcription_output,
            sentiment_output,
            score_output,
            details_output
        ]
    )


if __name__ == "__main__":
    demo.launch(share=True)