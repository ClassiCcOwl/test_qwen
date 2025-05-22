FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV HOST=0.0.0.0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    git-lfs \
    build-essential \
    gcc && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct /app/Qwen2.5-VL-7B-Instruct

COPY requirements.txt .


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128


COPY flow.py .

EXPOSE 8000

CMD ["uvicorn", "flow:app", "--host", "0.0.0.0", "--port", "8000"]
