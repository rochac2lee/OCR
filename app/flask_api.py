# API Flask pra detectar números de camisa
# usa PaddleOCR pq roda rápido em CPU
from flask import Flask, request, jsonify
import numpy as np
import cv2
from .ocr import extract_jersey_numbers

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    # só pra ver se tá rodando
    return jsonify({
        "status": "ok",
        "message": "API de detecção de números de camisas ativa",
        "version": "1.0.0"
    })

@app.route("/predict", methods=["POST"])
def predict():
    # recebe imagem e retorna os números detectados
    # envia POST com form-data: image=arquivo.jpg
    
    # valida se mandou a imagem
    if "image" not in request.files:
        return jsonify({
            "error": "Campo 'image' é obrigatório",
            "detail": "Envie a imagem usando multipart/form-data com campo 'image'"
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "error": "Arquivo não enviado",
            "detail": "O campo 'image' está vazio"
        }), 400

    # checa se é jpg/png/webp (mimetype nem sempre vem)
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    filename_lower = file.filename.lower()
    
    if not any(filename_lower.endswith(ext) for ext in allowed_extensions):
        return jsonify({
            "error": "Formato de imagem não suportado",
            "detail": f"Use arquivos .jpg, .jpeg, .png ou .webp",
            "filename": file.filename
        }), 400

    # decodifica a imagem
    try:
        data = file.read()
        npimg = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                "error": "Não foi possível decodificar a imagem",
                "detail": "Arquivo de imagem corrompido ou inválido"
            }), 400
            
    except Exception as e:
        return jsonify({
            "error": "Erro ao processar imagem",
            "detail": str(e)
        }), 500

    # roda o OCR
    try:
        results = extract_jersey_numbers(img)
        
        # formata a resposta com accuracy em porcentagem
        formatted_results = [
            {
                "number": r["number"],
                "accuracy": f"{round(float(r['confidence'])*100, 1)}%",
            }
            for r in results
        ]
        
        
        return jsonify({
            "results": formatted_results,
            "count": len(formatted_results)
        })
        
    except Exception as e:
        return jsonify({
            "error": "Erro ao extrair números",
            "detail": str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "Arquivo muito grande",
        "detail": "O tamanho máximo permitido é 16MB"
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "error": "Erro interno do servidor",
        "detail": "Ocorreu um erro inesperado"
    }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
