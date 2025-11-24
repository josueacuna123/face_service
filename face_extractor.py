from deepface import DeepFace
import numpy as np

class FaceExtractor:

    def extract_embedding(self, img):
        try:
            embedding = DeepFace.represent(
                img_path=img,
                model_name="Facenet512",
                enforce_detection=True
            )

            # Asegurar que siempre regresamos un vector numpy
            return np.array(embedding[0]["embedding"], dtype=float)

        except Exception as e:
            print("Error extrayendo embedding:", e)
            return None
