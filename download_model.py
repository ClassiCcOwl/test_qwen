from huggingface_hub import snapshot_download

# Download the model to the specified directory
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct", local_dir="/app/Qwen2.5-VL-7B-Instruct"
)
