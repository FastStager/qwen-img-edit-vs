from huggingface_hub import snapshot_download
import os

cache_dir = "/app/cache"

def ensure_model(repo_id: str):
    target_dir = os.path.join(cache_dir, repo_id.replace("/", "--"))
    if os.path.exists(target_dir):
        print(f"✅ {repo_id} already cached at {target_dir}")
        return
    print(f"⬇️ downloading {repo_id}...")
    snapshot_download(repo_id=repo_id, cache_dir=cache_dir, max_workers=8)
    print(f"✅ finished downloading {repo_id}")

def warmup_all():
    print("starting model warmup...")
    ensure_model("Qwen/Qwen-Image-Edit")
    ensure_model("lightx2v/Qwen-Image-Lightning")
    print("all models ready")

if __name__ == "__main__":
    warmup_all()
