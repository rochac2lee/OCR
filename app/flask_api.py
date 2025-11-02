from flask import Flask, request, jsonify
import numpy as np
import cv2
from .ocr import extract_jersey_numbers

app = Flask(__name__)

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"detail": "Campo 'image' é obrigatório"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"detail": "Arquivo não enviado"}), 400

    # Opcional: validar mimetype
    if file.mimetype not in ("image/jpeg", "image/png", "image/webp"):
        return jsonify({"detail": "Formato de imagem não suportado. Use JPEG, PNG ou WEBP."}), 400

    data = file.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"detail": "Não foi possível decodificar a imagem enviada."}), 400

    results = extract_jersey_numbers(img)
    # Formatar resposta: remover bbox, renomear confidence para accuracy em %
    formatted_results = [
        {
            "number": r["number"],
            "accuracy": round(r["confidence"] * 100),
        }
        for r in results
    ]
    return jsonify({"results": formatted_results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
