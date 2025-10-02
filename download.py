from huggingface_hub import snapshot_download

print("Starting CPU-safe model download phase...")
cache_dir = "/app/cache"

print("Downloading Qwen/Qwen-Image-Edit-2509...")
snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit-2509",
    cache_dir=cache_dir,
    local_files_only=False,
    max_workers=8,
)

print("Downloading lightx2v/Qwen-Image-Lightning LoRA...")
snapshot_download(
    repo_id="lightx2v/Qwen-Image-Lightning",
    cache_dir=cache_dir,
    local_files_only=False,
    max_workers=8,
)

print("All models downloaded to /app/cache. Ready for GPU compilation phase.")