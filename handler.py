import runpod
import torch
import numpy as np
from PIL import Image
import base64, io, os, math

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwen_image_edit import QwenImageEditPipeline as QwenImageEditPipelineCustom
from optimization import optimize_pipeline_
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
from download import warmup_all  # lazy model download

pipe = None
COMPILED_MODEL_PATH = "compiled_pipe.pt"

def load_model():
    global pipe
    if pipe is not None:
        return pipe

    warmup_all()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(COMPILED_MODEL_PATH):
        print(f"loading compiled model from {COMPILED_MODEL_PATH} to {device}...")
        pipe = torch.load(COMPILED_MODEL_PATH)
        pipe.to(device)
        return pipe

    print("compiled model not found, starting one-time compilation...")
    pipe = load_and_compile_model()
    return pipe

def load_and_compile_model():
    dtype = torch.bfloat16
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")

    scheduler = FlowMatchEulerDiscreteScheduler.from_config({
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
    })

    print("loading base pipeline...")
    compiled_pipe = QwenImageEditPipelineCustom.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        scheduler=scheduler,
        torch_dtype=dtype,
        cache_dir="/app/cache"
    ).to(device)

    compiled_pipe.transformer.__class__ = QwenImageTransformer2DModel
    compiled_pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

    try:
        print("loading and fusing LoRA...")
        compiled_pipe.load_lora_weights(
            "/app/cache/hub",
            weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
        )
        compiled_pipe.fuse_lora()
        print("LoRA fused")
    except Exception as e:
        print(f"LoRA load failed: {e}")

    try:
        print("running AOT compilation...")
        optimize_pipeline_(compiled_pipe, image=Image.new("RGB", (1024, 1024)), prompt="a cat")
        print("AOT compile ok")
    except Exception as e:
        print(f"AOT compile failed: {e}")

    return compiled_pipe

def base64_to_pil(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(job):
    global pipe
    if pipe is None:
        pipe = load_model()

    job_input = job["input"]
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "missing 'image' key in input"}

    prompt = job_input.get("prompt", "make it beautiful")
    seed = job_input.get("seed") or np.random.randint(0, np.iinfo(np.int32).max)
    true_guidance_scale = float(job_input.get("true_guidance_scale", 4.0))
    num_inference_steps = int(job_input.get("num_inference_steps", 8))

    generator = torch.Generator("cuda").manual_seed(seed)
    input_image = base64_to_pil(image_b64)

    suffix = "Strictly preserve all unmentioned objects and composition."
    final_prompt = f"{prompt}. {suffix}"

    try:
        output = pipe(
            image=input_image,
            prompt=final_prompt,
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=1,
        ).images[0]

        return {"image": pil_to_base64(output), "seed": seed, "version": "1.0"}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
