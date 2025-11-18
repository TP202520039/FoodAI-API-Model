from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI
import numpy as np
import requests
from pydantic import BaseModel

app = FastAPI()

# ------------------------------
# 1) Cargar Modelo
# ------------------------------
MODEL_PATH = "best_model_finetunedv1.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['ADOBO AREQUIPENO', 'AGUADITO DE POLLO', 'AJI DE GALLINA', 'ANTICUCHOS', 'ARROZ CON LECHE', 'ARROZ CON POLLO', 'BROASTER', 'CABRITO A LA NORTENA', 'CALDO DE RES', 'CARAPULCRA', 'CAU CAU', 'CAUSA', 'CAUSA FERRENAFANA', 'CEVICHE', 'CHANFAINITA', 'CHICHARRON', 'CHORITOS A LA CHALACA', 'CHUPE DE CAMARONES', 'CUY CHACTADO', 'ESTOFADO', 'JUANE', 'LOCRO', 'LOMO SALTADO', 'MAZAMORRA MORADA', 'MENESTRON', 'OLLUQUITO', 'PACHAMANCA', 'PAPA A LA HUANCAINA', 'PAPA RELLENA', 'PARIHUELA', 'PATASCA', 'PICANTE DE MARISCOS', 'PICARONES', 'POLLO A LA OLLA', 'POLLO AL HORNO', 'POLLO AL SILLAO', 'ROCOTO RELLENO', 'SALCHIPAPAS', 'SANCOCHADO', 'SECO', 'SHAMBAR', 'SOPA A LA MINUTA', 'SOPA CRIOLLA', 'SUDADO', 'TACACHO CON CECINA', 'TACU TACU', 'TALLARIN SALTADO', 'TALLARINES ROJOS', 'TALLARINES VERDES', 'TRUCHA FRITA']


def predecir_imagen(img, model, class_names, img_size=(224,224)):
    # Convertir a RGB (por si la imagen tiene transparencia/RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Redimensionar imagen
    img = img.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)   # (1,224,224,3)

    # Preprocesar igual que EfficientNet
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    # Predicción
    preds = model.predict(img_array, verbose=0)[0]

    # Top 1 clase y probabilidad
    top1_idx = np.argmax(preds)
    top1_clase = class_names[top1_idx]
    top1_prob = float(preds[top1_idx])

    return top1_clase, top1_prob


class ImageUrlRequest(BaseModel):
    imageUrl: str

@app.post("/detect-food")
async def predict(request: ImageUrlRequest):

    response = requests.get(request.imageUrl)
    img = Image.open(BytesIO(response.content))
    
    clase, confidence = predecir_imagen(img, model, class_names)

    # Si no detecta con una confianza superior al 0.6, devolver un arreglo vacio
    if confidence < 0.6:
        return {
            "detected_foods": []
        }

    return {
        "detected_foods" : [
            {
                "name": clase.lower(),
                "confidence":  int(confidence * 100) / 100
            }
        ]
    }

    