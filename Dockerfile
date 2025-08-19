FROM python:3.13-slim

# Εγκατάσταση system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Δημιουργία user
RUN useradd -m -u 1000 user

WORKDIR /code

# Αντιγραφή requirements
COPY requirements.txt /code/requirements.txt

# Εγκατάσταση Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Αντιγραφή κώδικα
COPY --chown=user . /code

# Δημιουργία directories
RUN mkdir -p data/videos/app_demo results temp && \
    chown -R user:user /code

USER user

# Environment variables
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
