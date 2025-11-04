# Jersey Number OCR

API REST para detectar números em camisas de futebol usando PaddleOCR.

## Como funciona

O sistema aplica várias técnicas de pré-processamento na imagem (CLAHE, operações morfológicas, adaptive thresholding, upscaling) antes de passar pelo OCR. Isso aumenta bastante a taxa de detecção, especialmente em imagens com baixa qualidade ou números pequenos.

Otimizado para rodar em CPU - os testes mostraram que usar 1 thread é mais rápido que multithreading nesse caso.

## Resultados
![ezgif-2c753fd31b4a01](https://github.com/user-attachments/assets/f5c6698b-29e4-4de2-9e46-50ce68ebd96d)


## Rodando com Docker

```bash
git clone https://github.com/rochac2lee/jersey-number-ocr
cd ocr
docker compose up --build
```

API vai estar em `http://localhost:8000`

## Usando a API

### Detectar números

```bash
curl -X POST http://localhost:8000/predict -F "image=@camisa.jpg"
```

Resposta:
```json
{
  "results": [
    {
      "number": "10",
      "accuracy": "95.3%"
    }
  ],
  "count": 1
}
```

Aceita JPG, PNG e WEBP.

### Exemplo em Python

```python
import requests

files = {"image": open("camisa.jpg", "rb")}
response = requests.post("http://localhost:8000/predict", files=files)
print(response.json())
```

## Estrutura do projeto

```
app/
├── flask_api.py    # rotas da API
└── ocr.py          # lógica do OCR e processamento de imagem

docker-compose.yml
Dockerfile
requirements.txt
```

## Rodando local (sem Docker)

```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export FLASK_APP=app.flask_api:app
flask run --host 0.0.0.0 --port 8000
```

## Stack

- Flask 3.0.3
- PaddleOCR 2.7.3
- OpenCV 4.10
- NumPy 1.24.3

## Performance

Tempo médio: 1-3 segundos por imagem, dependendo do tamanho e da quantidade de números.

## Limpando o ambiente Docker

```bash
docker compose down
docker system prune -a --volumes -f
```

## Licença

Esse projeto é distribuído sob a licença [MIT](LICENSE)
