# Usa imagem base Amazon Linux 2023 (mais leve)
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Variáveis de ambiente para otimização
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8

# Instala Python 3.9 e dependências essenciais
RUN dnf -y update && \
    dnf -y install \
      python3.9 \
      python3.9-devel \
      gcc \
      gcc-c++ \
      make \
      glib2 \
      mesa-libGL \
      wget \
      && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Cria e ativa ambiente virtual Python
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Copia e instala dependências Python (cached layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# ------------------------------------------------------
# PATCHES para compatibilidade PaddleX com Python 3.9
# ------------------------------------------------------
# Corrige sintaxes Python 3.10+ que não funcionam em 3.9
RUN if [ -f /opt/venv/lib64/python3.9/site-packages/paddlex/inference/pipelines/paddleocr_vl/uilts.py ]; then \
      sed -i '1a\from typing import Optional' \
        /opt/venv/lib64/python3.9/site-packages/paddlex/inference/pipelines/paddleocr_vl/uilts.py && \
      sed -i 's/def find_shortest_repeating_substring(s: str) -> str | None:/def find_shortest_repeating_substring(s: str) -> Optional[str]:/g' \
        /opt/venv/lib64/python3.9/site-packages/paddlex/inference/pipelines/paddleocr_vl/uilts.py && \
      sed -i 's/) -> Tuple\[str, str, int\] | None:/) -> Optional[Tuple[str, str, int]]:/g' \
        /opt/venv/lib64/python3.9/site-packages/paddlex/inference/pipelines/paddleocr_vl/uilts.py; \
    fi

# ------------------------------------------------------
# PRÉ-DOWNLOAD DOS MODELOS PADDLEOCR
# Baixa modelos durante o build para evitar download no runtime
# ------------------------------------------------------
RUN python3.9 -c "from paddleocr import PaddleOCR; \
    print('Iniciando download dos modelos PaddleOCR...'); \
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=True); \
    print('Modelos PaddleOCR baixados com sucesso!')" || true

# Copia código da aplicação
COPY app /app/app

# Cria diretórios necessários
RUN mkdir -p /app/logs /app/tmp_output

# Expõe porta da API
EXPOSE 8000

# Comando para iniciar a aplicação com hot reload
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8000", "--reload"]
