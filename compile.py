import torch
import math
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwen_image_edit import QwenImageEditPipeline as QwenImageEditPipelineCustom
from optimization import optimize_pipeline_

print("Starting GPU-powered AOT compilation phase...")

def load_base_pipeline():
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for compilation.")

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

    pipe = QwenImageEditPipelineCustom.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        scheduler=scheduler,
        torch_dtype=dtype,
        cache_dir="/app/cache"
    ).to(device)

    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
    from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
    pipe.transformer.__class__ = QwenImageTransformer2DModel
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

    try:
        pipe.load_lora_weights(
            "/app/cache/hub",
            weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
        )
        pipe.fuse_lora()
        print("LoRA weights fused successfully.")
    except Exception as e:
        print(f"Could not load LoRA: {e}")
        
    return pipe

pipe = load_base_pipeline()

print("Compiling the transformer model... This will take several minutes.")
try:
    optimize_pipeline_(pipe, image=Image.new("RGB", (1024, 1024)), prompt="a cat")
    print("AOT compilation successful.")
except Exception as e:
    print(f"AOT compile failed: {e}")
    
COMPILED_MODEL_PATH = "compiled_pipe.pt"
print(f"Saving compiled pipeline to {COMPILED_MODEL_PATH}...")
torch.save(pipe, COMPILED_MODEL_PATH)
print("Compilation and saving complete. The worker is ready for runtime.")