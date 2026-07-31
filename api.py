# ============================================================
# API.PY
# Auteure : Adji Fatou NGOM
# Description : API REST avec FastAPI
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pipeline import run_pipeline

app = FastAPI(
    title       = "API Sentiment Audio",
    description = "Pipeline Audio vers Sentiment avec Wav2Vec 2.0 et BERT",
    version     = "1.0.0"
)


@app.get("/")
def root():
    """Retourne les informations de l API."""
    return {
        "message"   : "API Sentiment Audio",
        "version"   : "1.0.0",
        "endpoints" : {
            "POST /predict" : "Analyser le sentiment d un fichier audio"
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Analyse le sentiment d un fichier audio uploade."""
    if not file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(
            status_code = 400,
            detail      = "Format non supporte. Utilisez .wav ou .mp3"
        )
    with tempfile.NamedTemporaryFile(
        delete = False,
        suffix = os.path.splitext(file.filename)[1]
    ) as tmp:
        content  = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        resultat = run_pipeline(tmp_path)
        return JSONResponse(content=resultat)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)