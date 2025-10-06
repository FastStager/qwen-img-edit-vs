import runpod
import torch
import numpy as np
from PIL import Image
import base64
import io
import os
import math

from huggingface_hub import snapshot_download

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from optimization import optimize_pipeline_
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

pipe = None
COMPILED_MODEL_PATH = "compiled_pipe.pt"

def download_models():
    """Downloads models from Hugging Face Hub."""
    cache_dir = "/app/cache"
    print("Downloading Qwen/Qwen-Image-Edit-2509...")
    snapshot_download(repo_id="Qwen/Qwen-Image-Edit-2509", cache_dir=cache_dir, max_workers=8)
    print("Downloading lightx2v/Qwen-Image-Lightning LoRA...")
    snapshot_download(repo_id="lightx2v/Qwen-Image-Lightning", cache_dir=cache_dir, max_workers=8)
    print("All models downloaded.")

def load_and_compile_model():
    """
    Downloads, loads, and compiles the model. Run only on the first start.
    """
    download_models()
    dtype = torch.bfloat16
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for compilation and inference.")

    print("Loading base model...")
    scheduler_config = {
        "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
        "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
        "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
        "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
        "use_exponential_sigmas": False, "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    # Using the correct class name here
    compiled_pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        scheduler=scheduler,
        torch_dtype=dtype,
        cache_dir="/app/cache"
    ).to(device)

    compiled_pipe.transformer.__class__ = QwenImageTransformer2DModel
    compiled_pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

    try:
        print("Loading and fusing LoRA weights...")
        compiled_pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safensors",
            cache_dir="/app/cache"
        )
        compiled_pipe.fuse_lora()
        print("LoRA weights fused successfully.")
    except Exception as e:
        print(f"Could not load LoRA: {e}")

    print("Compiling the transformer model...")
    try:
        dummy_images = [Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))]
        optimize_pipeline_(compiled_pipe, image=dummy_images, prompt="a cat")
        print("AOT compilation successful.")
    except Exception as e:
        print(f"AOT compile failed: {e}")
        
    return compiled_pipe

def load_model():
    """
    Loads the pipeline. If a pre-compiled version exists, it loads from disk.
    Otherwise, it downloads, compiles, and saves the model for future runs.
    """
    global pipe
    if pipe is not None:
        return pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(COMPILED_MODEL_PATH):
        print(f"Loading compiled model from {COMPILED_MODEL_PATH} to {device}...")
        pipe = torch.load(COMPILED_MODEL_PATH)
        pipe.to(device)
        print("Model loaded successfully from disk.")
    else:
        print("Compiled model not found. Starting one-time setup...")
        pipe = load_and_compile_model()
        print(f"Saving compiled pipeline to {COMPILED_MODEL_PATH} for future runs...")
        torch.save(pipe, COMPILED_MODEL_PATH)
        print("First-time setup complete.")
    return pipe

def base64_to_pil(base64_string):
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handler(job):
    global pipe
    if pipe is None:
        load_model()
    job_input = job['input']
    images_b64 = job_input.get('images', [])
    if not isinstance(images_b64, list):
         return {"error": "'images' must be a list of base64-encoded strings."}
    if not images_b64 and 'image' in job_input:
        images_b64 = [job_input.get('image')]
    prompt = job_input.get('prompt', 'make it beautiful')
    seed = job_input.get('seed', None)
    true_guidance_scale = float(job_input.get('true_guidance_scale', 1.0))
    num_inference_steps = int(job_input.get('num_inference_steps', 8))
    num_outputs = int(job_input.get('num_outputs', 1))
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input_images = [base64_to_pil(img_b64) for img_b64 in images_b64]
    print(f"Processing job with {len(input_images)} images, seed: {seed}, steps: {num_inference_steps}, guidance: {true_guidance_scale}")
    print(f"Prompt: {prompt}")
    try:
        output_images = pipe(
            image=input_images if input_images else None,
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_outputs,
        ).images
        output_b64_list = [pil_to_base64(img) for img in output_images]
        result = {"images": output_b64_list, "seed": seed, "version": "2.0"}
        if len(output_b64_list) == 1:
            result["image"] = output_b64_list[0]
        return result
    except Exception as e:
        print(f"Inference error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})