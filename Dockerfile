FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV HOST=0.0.0.0

RUN apt-get update && \
    apt-get install -y build-essential gcc && \
    apt-get install -y libgl1 git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct /app/Qwen2.5-VL-7B-Instruct

COPY flow.py ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    accelerate \
    python-multipart \
    uvicorn \
    pillow \
    'qwen-vl-utils[decord]==0.0.8' && \
    pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
#     pip install autoawq

# git+https://github.com/huggingface/transformers.git@8ee50537fe7613b87881cd043a85971c85e99519
RUN pip install git+https://github.com/huggingface/transformers.git@tiny-fixes-qwen2.5-vl

EXPOSE 8000

CMD ["uvicorn", "flow:app", "--host", "0.0.0.0", "--port", "8000"]
