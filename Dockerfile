FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# dependencias sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# primero pip base (CRÍTICO)
RUN pip install --upgrade pip

# copia requirements
COPY requirements.txt .

# instala setuptools ANTES (clave para pkg_resources)
RUN pip install setuptools wheel

# instala dependencias python
RUN pip install --no-cache-dir -r requirements.txt

# copia código
COPY . .

ENV PORT=10000

# Streamlit correcto
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
