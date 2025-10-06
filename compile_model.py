import torch
from PIL import Image
import math
import os

from huggingface_hub import snapshot_download
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from optimization import optimize_pipeline_
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

COMPILED_MODEL_PATH = "compiled_pipe.pt"

def download_models():
    """Downloads models from Hugging Face Hub."""
    cache_dir = "/app/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        
    print("Downloading Qwen/Qwen-Image-Edit-2509...")
    snapshot_download(repo_id="Qwen/Qwen-Image-Edit-2509", cache_dir=cache_dir, max_workers=8)
    print("Downloading lightx2v/Qwen-Image-Lightning LoRA...")
    snapshot_download(repo_id="lightx2v/Qwen-Image-Lightning", cache_dir=cache_dir, max_workers=8)
    print("All models downloaded.")

def main():
    """
    Downloads, loads, compiles the model, and saves the compiled artifact.
    """
    print("Starting one-time model compilation...")
    
    os.environ['HF_HOME'] = '/app/cache'
    os.environ['HUGGING_FACE_HUB_CACHE'] = '/app/cache'

    download_models()
    
    dtype = torch.bfloat16
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for this compilation step.")

    print("Loading base model...")
    scheduler_config = {
        "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
        "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
        "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
        "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
        "use_exponential_sigmas": False, "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        scheduler=scheduler,
        torch_dtype=dtype,
        cache_dir="/app/cache"
    ).to(device)

    pipe.transformer.__class__ = QwenImageTransformer2DModel
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

    try:
        print("Loading and fusing LoRA weights...")
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safensors",
            cache_dir="/app/cache"
        )
        pipe.fuse_lora()
    except Exception as e:
        print(f"Could not load LoRA: {e}")

    print("Compiling the transformer model (this will take several minutes)...")
    try:
        dummy_images = [Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))]
        optimize_pipeline_(pipe, image=dummy_images, prompt="a cat")
        print("AOT compilation successful.")
    except Exception as e:
        print(f"AOT compile failed: {e}")

    print(f"Saving compiled pipeline to {COMPILED_MODEL_PATH}...")
    torch.save(pipe, COMPILED_MODEL_PATH)
    print("Compilation complete. Artifact saved.")

if __name__ == "__main__":
    main()