"""
API Flask para extração de números de camisas de atletas.
Otimizado para máxima velocidade em CPU usando PaddleOCR.
"""
from flask import Flask, request, jsonify
import numpy as np
import cv2
import time
from .ocr import extract_jersey_numbers

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "message": "API de detecção de números de camisas ativa",
        "version": "1.0.0"
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principal para detecção de números em camisas de atletas.
    
    Expects:
        - Multipart form-data com campo 'image'
        - Formatos suportados: JPEG, PNG, WEBP
        
    Returns:
        JSON com lista de números detectados e suas precisões (0-100%)
        
    Example:
        curl -X POST -F "image=@camisa.jpg" http://localhost:8000/predict
    """
    start_time = time.time()
    
    # Validação de entrada
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

    # Validar formato de imagem por extensão (mimetype pode estar vazio)
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    filename_lower = file.filename.lower()
    
    if not any(filename_lower.endswith(ext) for ext in allowed_extensions):
        return jsonify({
            "error": "Formato de imagem não suportado",
            "detail": f"Use arquivos .jpg, .jpeg, .png ou .webp",
            "filename": file.filename
        }), 400

    # Processar imagem
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

    # Extrair números
    try:
        results = extract_jersey_numbers(img)
        
        # Formatar resposta: remover bbox, renomear confidence para accuracy em %
        formatted_results = [
            {
                "number": r["number"],
                "accuracy": round(r["confidence"] * 100),
            }
            for r in results
        ]
        
        processing_time = round((time.time() - start_time) * 1000, 2)  # ms
        
        return jsonify({
            "success": True,
            "results": formatted_results,
            "count": len(formatted_results),
            "processing_time_ms": processing_time
        })
        
    except Exception as e:
        return jsonify({
            "error": "Erro ao extrair números",
            "detail": str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle payload too large errors."""
    return jsonify({
        "error": "Arquivo muito grande",
        "detail": "O tamanho máximo permitido é 16MB"
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    return jsonify({
        "error": "Erro interno do servidor",
        "detail": "Ocorreu um erro inesperado"
    }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
