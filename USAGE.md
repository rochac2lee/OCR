# Guia de Uso - API de Detecção de Números

## 🚀 Quick Start

### 1. Iniciar a API

```bash
# Build e start
docker-compose up --build

# Ou em background
docker-compose up -d --build
```

Aguarde a mensagem: `"PaddleOCR inicializado com sucesso"`

### 2. Testar

```bash
# Health check
curl http://localhost:8000/

# Enviar imagem
curl -X POST -F "image=@camisa.jpg" http://localhost:8000/predict
```

## 📸 Exemplos de Uso

### Python

```python
import requests

# Enviar imagem
url = "http://localhost:8000/predict"
files = {"image": open("camisa.jpg", "rb")}

response = requests.post(url, files=files)
data = response.json()

print(f"Sucesso: {data['success']}")
print(f"Números detectados: {data['count']}")
for result in data['results']:
    print(f"  - Número: {result['number']} (Precisão: {result['accuracy']}%)")
print(f"Tempo: {data['processing_time_ms']}ms")
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function detectNumber(imagePath) {
  const form = new FormData();
  form.append('image', fs.createReadStream(imagePath));

  const response = await axios.post('http://localhost:8000/predict', form, {
    headers: form.getHeaders()
  });

  console.log('Resultados:', response.data);
  return response.data;
}

detectNumber('./camisa.jpg');
```

### cURL

```bash
# Detecção básica
curl -X POST \
  -F "image=@camisa.jpg" \
  http://localhost:8000/predict

# Com formatação JSON
curl -X POST \
  -F "image=@camisa.jpg" \
  http://localhost:8000/predict | jq .

# Salvar resposta em arquivo
curl -X POST \
  -F "image=@camisa.jpg" \
  http://localhost:8000/predict > resultado.json
```

### Postman

1. Método: `POST`
2. URL: `http://localhost:8000/predict`
3. Body: `form-data`
4. Key: `image` (tipo: File)
5. Value: Selecione sua imagem

## 🔍 Formato de Resposta

### Sucesso

```json
{
  "success": true,
  "results": [
    {
      "number": "10",
      "accuracy": 95
    },
    {
      "number": "7",
      "accuracy": 88
    }
  ],
  "count": 2,
  "processing_time_ms": 234.5
}
```

### Erro

```json
{
  "error": "Campo 'image' é obrigatório",
  "detail": "Envie a imagem usando multipart/form-data com campo 'image'"
}
```

## 🎯 Dicas para Melhores Resultados

### ✅ Boas Práticas

- Use imagens com boa iluminação
- Números devem estar visíveis e legíveis
- Prefira imagens com fundo contrastante
- Tamanho recomendado: 640x480 a 1920x1080
- Formatos: JPEG, PNG ou WEBP

### ⚠️ Evite

- Imagens muito pequenas (< 200x200)
- Imagens muito grandes (> 4000x4000)
- Números muito distorcidos ou borrados
- Baixíssimo contraste entre número e fundo

## 📊 Interpretação da Precisão (Accuracy)

- **90-100%**: Detecção muito confiável
- **75-89%**: Detecção confiável
- **60-74%**: Detecção razoável (revisar)
- **< 60%**: Detecção incerta (filtrado automaticamente)

## 🛠️ Desenvolvimento

### Modificar Código

1. Edite arquivos em `app/`
2. Flask recarrega automaticamente (hot reload)
3. Teste novamente

### Ver Logs

```bash
# Logs em tempo real
docker-compose logs -f

# Logs da API apenas
docker-compose logs -f api

# Últimas 100 linhas
docker-compose logs --tail=100
```

### Reiniciar Serviço

```bash
# Reinício rápido (sem rebuild)
docker-compose restart

# Rebuild completo
docker-compose down
docker-compose up --build
```

### Parar Serviço

```bash
# Parar
docker-compose down

# Parar e limpar tudo
docker-compose down
docker system prune -a --volumes -f
```

## 🔧 Troubleshooting

### Erro: "Connection refused"
```bash
# Verificar se o container está rodando
docker ps

# Verificar logs
docker-compose logs
```

### Erro: "Out of memory"
```bash
# Aumentar limite de memória no docker-compose.yml
mem_limit: 4g
```

### Detecção lenta
- Primeira detecção é mais lenta (carrega modelos)
- Detecções subsequentes são rápidas (150-300ms)

### Números não detectados
- Verifique qualidade da imagem
- Tente com boa iluminação
- Certifique-se de que números estão visíveis

## 📈 Performance Esperada

- **Primeira requisição**: 1-3 segundos (carrega modelos)
- **Requisições subsequentes**: 150-300ms
- **Uso de memória**: ~1.5GB
- **Uso de CPU**: 1-2 cores durante processamento

## 🎓 Exemplos Avançados

### Processar múltiplas imagens

```python
import requests
import os
from pathlib import Path

def processar_lote(diretorio):
    url = "http://localhost:8000/predict"
    resultados = []
    
    for img_path in Path(diretorio).glob("*.jpg"):
        with open(img_path, "rb") as f:
            files = {"image": f}
            response = requests.post(url, files=files)
            data = response.json()
            
            resultados.append({
                "arquivo": img_path.name,
                "numeros": [r["number"] for r in data.get("results", [])],
                "tempo_ms": data.get("processing_time_ms")
            })
    
    return resultados

# Usar
resultados = processar_lote("./imagens")
for r in resultados:
    print(f"{r['arquivo']}: {r['numeros']} ({r['tempo_ms']}ms)")
```

### Integração com OpenCV

```python
import cv2
import requests
import numpy as np

def detectar_de_video(video_path, intervalo_frames=30):
    """Detecta números a cada N frames de um vídeo"""
    cap = cv2.VideoCapture(video_path)
    url = "http://localhost:8000/predict"
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % intervalo_frames == 0:
            # Converte frame para bytes
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # Envia para API
            files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Frame {frame_count}: {data.get('results', [])}")
        
        frame_count += 1
    
    cap.release()

# Usar
detectar_de_video("jogo.mp4", intervalo_frames=30)
```

---

**Precisa de ajuda?** Verifique os logs com `docker-compose logs -f`

