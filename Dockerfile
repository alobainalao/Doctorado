# Imagen base con Python 3.10 (IMPORTANTE para tu error)
FROM python:3.10-slim

# Evita problemas de compilación
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema (importante para scipy, numpy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (mejor cache)
COPY requirements.txt .

# Actualizar pip
RUN pip install --upgrade pip setuptools wheel

# Instalar dependencias Python
RUN pip install -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Puerto (Render lo usa)
ENV PORT=10000

# Comando de ejecución (AJUSTA ESTO)
CMD ["python", "app.py"]
