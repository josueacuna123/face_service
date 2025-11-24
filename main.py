from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from face_extractor import FaceExtractor
import uvicorn
import tempfile
import numpy as np
import os

app = FastAPI()
extractor = FaceExtractor()

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Obtener embedding
        embedding = extractor.extract_embedding(tmp_path)

        # Borrar archivo temporal
        os.remove(tmp_path)

        # Validar embedding
        if embedding is None:
            raise HTTPException(status_code=400, detail="No se pudo extraer embedding")

        # Ahora embedding SIEMPRE es un numpy array
        return JSONResponse(content={"embedding": embedding.tolist()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)