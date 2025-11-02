FROM amazonlinux:2023

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Instala Python 3.9 e dependências de sistema necessárias
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
      tar \
      gzip \
      which && \
    dnf clean all

# Cria e usa um venv em /opt/venv
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------
# PATCH 1: Injetar a importação 'Optional'
# ------------------------------------------------------
# Insere 'from typing import Optional' após a primeira linha do arquivo (1a).
RUN sed -i '1a\from typing import Optional' \
    /opt/venv/lib64/python3.9/site-packages/paddlex/inference/pipelines/paddleocr_vl/uilts.py

# ------------------------------------------------------
# PATCH 2: Corrigir TODAS as sintaxes 'TYPE | None' para Python 3.9
# Os colchetes [] são escapados com '\' para garantir a correspondência exata.
# ------------------------------------------------------
# 1. Corrige a função 'find_shortest_repeating_substring' (linha ~832)
RUN sed -i 's/def find_shortest_repeating_substring(s: str) -> str | None:/def find_shortest_repeating_substring(s: str) -> Optional[str]:/g' \
    /opt/venv/lib64/python3.9/site-packages/paddlex/inference/pipelines/paddleocr_vl/uilts.py

# 2. Corrige a função 'find_best_match' (linha ~854). ESTA É A CORREÇÃO CRÍTICA.
# Escapa os colchetes [] no padrão de busca.
RUN sed -i 's/) -> Tuple\[str, str, int\] | None:/) -> Optional[Tuple[str, str, int]]:/g' \
    /opt/venv/lib64/python3.9/site-packages/paddlex/inference/pipelines/paddleocr_vl/uilts.py

# ------------------------------------------------------

COPY app /app/app

EXPOSE 8000

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8000", "--reload"]
