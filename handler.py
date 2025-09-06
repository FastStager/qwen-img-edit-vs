import runpod
import torch
import numpy as np
from PIL import Image
import base64
import io
import os
import math

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwen_image_edit import QwenImageEditPipeline as QwenImageEditPipelineCustom
from optimization import optimize_pipeline_
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

pipe = None
COMPILED_MODEL_PATH = "compiled_pipe.pt"

def load_model():
    """
    Loads the pipeline. If a pre-compiled version exists, it loads from disk.
    Otherwise, it compiles the model and saves it for future runs.
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
        print("Compiled model not found. Starting one-time compilation...")
        pipe = load_and_compile_model()
        print(f"Saving compiled pipeline to {COMPILED_MODEL_PATH} for future runs...")
        torch.save(pipe, COMPILED_MODEL_PATH)
        print("Compilation and saving complete.")

    return pipe

def load_and_compile_model():
    """
    Loads the base model, compiles it, and returns the pipeline.
    This is run only on the first start of a new worker.
    """
    dtype = torch.bfloat16
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for compilation and inference.")

    print("Loading base model for the first time...")
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    compiled_pipe = QwenImageEditPipelineCustom.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        scheduler=scheduler,
        torch_dtype=dtype,
        cache_dir="/app/cache"
    ).to(device)

    compiled_pipe.transformer.__class__ = QwenImageTransformer2DModel
    compiled_pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

    try:
        print("Loading and fusing LoRA weights...")
        compiled_pipe.load_lora_weights(
            "/app/cache/hub",
            weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
        )
        compiled_pipe.fuse_lora()
        print("LoRA weights fused successfully.")
    except Exception as e:
        print(f"Could not load LoRA: {e}")

    print("Compiling the transformer model... This will take several minutes.")
    try:
        optimize_pipeline_(compiled_pipe, image=Image.new("RGB", (1024, 1024)), prompt="a cat")
        print("AOT compilation successful.")
    except Exception as e:
        print(f"AOT compile failed: {e}")
        
    return compiled_pipe

def base64_to_pil(base64_string):
    """Decodes a base64 string into a PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_to_base64(pil_image):
    """Encodes a PIL Image into a base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handler(job):
    """
    RunPod serverless handler function.
    """
    global pipe
    if pipe is None:
        load_model()

    job_input = job['input']
    image_b64 = job_input.get('image')
    if not image_b64:
        return {"error": "Missing 'image' key in input. Please provide a base64-encoded image."}
        
    prompt = job_input.get('prompt', 'make it beautiful')
    seed = job_input.get('seed', None)
    true_guidance_scale = float(job_input.get('true_guidance_scale', 4.0))
    num_inference_steps = int(job_input.get('num_inference_steps', 8))

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    input_image = base64_to_pil(image_b64)

    PRESERVATION_PROMPT_SUFFIX = (
        "Strictly preserve all unmentioned objects, details, and the overall composition of the original image. "
        "This includes keeping background elements like doors, windows, and furniture exactly as they are."
    )
    final_prompt = f"{prompt}. {PRESERVATION_PROMPT_SUFFIX}"
    
    print(f"Processing job with seed: {seed}, steps: {num_inference_steps}, guidance: {true_guidance_scale}")
    print(f"Final prompt: {final_prompt}")

    try:
        output_image = pipe(
            image=input_image,
            prompt=final_prompt,
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=1,
        ).images[0]

        output_b64 = pil_to_base64(output_image)
        return {
            "image": output_b64,
            "seed": seed,
            "version": "1.0" 
        }

    except Exception as e:
        print(f"Inference error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})