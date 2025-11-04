# Jersey Number OCR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Flask 3.0.3](https://img.shields.io/badge/flask-3.0.3-green.svg)](https://flask.palletsprojects.com/)
[![PaddleOCR 2.7.3](https://img.shields.io/badge/paddleocr-2.7.3-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)

API REST para detecção automática de números em camisas esportivas usando OCR (Optical Character Recognition).

## Índice

* [Como Funciona](#como-funciona)
* [Resultados](#resultados)
* [Quick Start](#quick-start)
* [Uso da API](#uso-da-api)
  * [Endpoint de Detecção](#endpoint-de-detecção)
  * [Exemplos](#exemplos)
* [Estrutura do Projeto](#estrutura-do-projeto)
* [Stack Tecnológico](#stack-tecnológico)
* [Docker](#docker)
  * [Build e Execução](#build-e-execução)
  * [Comandos Úteis](#comandos-úteis)
* [Desenvolvimento](#desenvolvimento)
  * [Rodando Localmente](#rodando-localmente)
* [Performance](#performance)
* [Contribuindo](#contribuindo)
* [Licença](#licença)

## Como Funciona

O sistema aplica múltiplas técnicas de pré-processamento na imagem antes de executar o OCR:

- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) para equalização de contraste
- **Operações morfológicas** para realce de bordas e remoção de ruído
- **Adaptive thresholding** com estratégias Gaussian e Mean
- **Upscaling inteligente** (1.5x e 2.0x) para melhor detecção de números pequenos
- **Filtragem bilateral** para preservação de bordas

O processamento em múltiplas variantes da mesma imagem aumenta significativamente a taxa de detecção, especialmente em condições adversas (baixa qualidade, iluminação irregular, números pequenos).

Otimizado para rodar em CPU - benchmarks mostraram que usar 1 thread é mais eficiente que multithreading para este caso de uso.

## Resultados

![ezgif-2c753fd31b4a01](https://github.com/user-attachments/assets/f5c6698b-29e4-4de2-9e46-50ce68ebd96d)

## Instruções

```bash
git clone https://github.com/rochac2lee/jersey-number-ocr
cd jersey-number-ocr
docker compose up --build
```

A API estará disponível em `http://localhost:8000`

## Uso da API

### Endpoint de Detecção

```http
POST /predict
Content-Type: multipart/form-data
```

**Parâmetros:**
- `image`: Arquivo de imagem (JPG, JPEG, PNG, WEBP)

**Resposta de Sucesso:**
```json
{
  "results": [
    {
      "number": "10",
      "accuracy": "95.3%"
    },
    {
      "number": "7",
      "accuracy": "89.1%"
    }
  ],
  "count": 2
}
```

**Resposta de Erro:**
```json
{
  "error": "Campo 'image' é obrigatório",
  "detail": "Envie a imagem usando multipart/form-data com campo 'image'"
}
```

### Exemplos

**cURL:**
```bash
curl -X POST http://localhost:8000/predict -F "image=@camisa.jpg"
```

**Python:**
```python
import requests

files = {"image": open("camisa.jpg", "rb")}
response = requests.post("http://localhost:8000/predict", files=files)
print(response.json())
```

**JavaScript:**
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Estrutura do Projeto

```
ocr/
├── app/
│   ├── __init__.py          # Inicialização do módulo
│   ├── flask_api.py         # Rotas e handlers da API REST
│   └── ocr.py              # Engine OCR e processamento de imagem
├── logs/                    # Logs da aplicação
├── docker-compose.yml       # Configuração Docker Compose
├── Dockerfile              # Build da imagem Docker
├── requirements.txt        # Dependências Python
├── LICENSE                 # Licença MIT
└── README.md              # Documentação
```

## Stack Tecnológico

| Tecnologia | Versão | Propósito |
|-----------|--------|-----------|
| Flask | 3.0.3 | Framework web |
| PaddleOCR | 2.7.3 | Engine de OCR |
| PaddlePaddle | 2.6.1 | Framework de ML |
| OpenCV | 4.10.0 | Processamento de imagem |
| NumPy | 1.24.3 | Computação numérica |
| Pillow | 10.4.0 | Manipulação de imagem |
| Docker | - | Containerização |

## Docker

### Build e Execução

**Usando Docker Compose (recomendado):**

```bash
# Iniciar o container em modo detached
docker compose up -d --build
```

## Desenvolvimento

### Rodando Localmente

Para desenvolvimento local sem Docker:

```bash
# Criar ambiente virtual
python3.9 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Iniciar servidor
export FLASK_APP=app.flask_api:app
flask run --host 0.0.0.0 --port 8000 --reload
```

O projeto suporta **hot reload** - alterações em arquivos Python são detectadas automaticamente.

## Performance

- **Tempo médio de resposta**: 1-3 segundos por imagem
- **Uso de CPU**: Otimizado para 1 thread
- **Uso de memória**: ~500MB-1GB
- **Limite de memória**: 2GB (swap: 4GB)
- **Tamanho máximo de imagem**: 16MB

## Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'feat: adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

**Padrão de commits**: Seguimos [Conventional Commits](https://www.conventionalcommits.org/)

## Licença

Esse projeto é distribuído sob a licença [MIT](LICENSE)

---

Desenvolvido por [Cleber Lee](https://github.com/rochac2lee)
