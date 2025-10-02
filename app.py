import gradio as gr
import torch
import numpy as np
import spaces
import math
from PIL import Image
import random
import os
import base64
import json

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
from optimization import optimize_pipeline_
from huggingface_hub import InferenceClient


pipe = None

SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise and comprehensive**. Avoid overly long sentences and unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the main part of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the scene in the input images.  
- If multiple sub-images are to be generated, describe the content of each sub-image individually.  

## 2. Task-Type Handling Rules

### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Keep the original language of the text, and keep the capitalization.  
- Both adding new text and replacing existing text are text replacement tasks, For example:  
    - Replace "xx" to "yy"  
    - Replace the mask / bounding box to "yy"  
    - Replace the visual object to "yy"  
- Specify text position, color, and layout only if user has required.  
- If font is specified, keep the original language of the font.  

### 3. Human Editing Tasks
- Make the smallest changes to the given user's prompt.  
- If changes to background, action, expression, camera shot, or ambient lighting are required, please list each modification individually.
- **Edits to makeup or facial features / expression must be subtle, not exaggerated, and must preserve the subject's identity consistency.**
    > Original: "Add eyebrows to the face"  
    > Rewritten: "Slightly thicken the person's eyebrows with little change, look natural."

### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco style: flashing lights, disco ball, mirrored walls, vibrant colors"  
- For style reference, analyze the original image and extract key characteristics (color, composition, texture, lighting, artistic style, etc.), integrating them into the instruction.  
- **Colorization tasks (including old photo restoration) must use the fixed template:**  
  "Restore and colorize the old photo."  
- Clearly specify the object to be modified. For example:  
    > Original: Modify the subject in Picture 1 to match the style of Picture 2.  
    > Rewritten: Change the girl in Picture 1 to the ink-wash style of Picture 2 — rendered in black-and-white watercolor with soft color transitions.

### 5. Material Replacement
- Clearly specify the object and the material. For example: "Change the material of the apple to papercut style."
- For text material replacement, use the fixed template:
    "Change the material of text "xxxx" to laser style"

### 6. Logo/Pattern Editing
- Material replacement should preserve the original shape and structure as much as possible. For example:
   > Original: "Convert to sapphire material"  
   > Rewritten: "Convert the main subject in the image to sapphire material, preserving similar shape and structure"
- When migrating logos/patterns to new scenes, ensure shape and structure consistency. For example:
   > Original: "Migrate the logo in the image to a new scene"  
   > Rewritten: "Migrate the logo in the image to a new scene, preserving similar shape and structure"

### 7. Multi-Image Tasks
- Rewritten prompts must clearly point out which image's element is being modified. For example:  
    > Original: "Replace the subject of picture 1 with the subject of picture 2"  
    > Rewritten: "Replace the girl of picture 1 with the boy of picture 2, keeping picture 2's background unchanged"  
- For stylization tasks, describe the reference image's style in the rewritten prompt, while preserving the visual content of the source image.  

## 3. Rationale and Logic Check
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" requires logical correction.
- Supplement missing critical information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, blank space, center/edge, etc.).

# Output Format Example
```json
{
   "Rewritten": "..."
}
'''

def encode_image(pil_image):
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def polish_prompt_hf(prompt, img_list):
    """
    Rewrites the prompt using a Hugging Face InferenceClient.
    """
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Warning: HF_TOKEN not set. Falling back to original prompt.")
        return prompt

    try:
        user_prompt_with_context = f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
        client = InferenceClient(provider="cerebras", api_key=api_key)

        messages = [
            {"role": "system", "content": "you are a helpful assistant, you should provide useful answers to users."},
            {"role": "user", "content": []}
        ]
        for img in img_list:
            messages[1]["content"].append({"image": f"data:image/png;base64,{encode_image(img)}"})
        messages[1]["content"].append({"text": user_prompt_with_context})

        completion = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            messages=messages,
        )
        
        result = completion.choices[0].message.content
        
        if '{"Rewritten"' in result:
            try:
                result = result.replace('```json', '').replace('```', '')
                result_json = json.loads(result)
                polished_prompt = result_json.get('Rewritten', result)
            except Exception:
                polished_prompt = result
        else:
            polished_prompt = result
            
        return polished_prompt.strip().replace("\n", " ")
    except Exception as e:
        print(f"Error during API call to Hugging Face: {e}")
        return prompt

def load_pipeline():
    """
    Loads and configures the Qwen-Image-Edit-Plus pipeline on GPU with custom scheduler, transformer, LoRA weights, and optimization.
    """
    global pipe
    if pipe is not None:
        return pipe

    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for inference.")

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

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        scheduler=scheduler,
        torch_dtype=dtype,
        cache_dir="/app/cache"
    ).to(device)

    pipe.transformer.__class__ = QwenImageTransformer2DModel
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

    try:
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
            cache_dir="/app/cache"
        )
        pipe.fuse_lora()
    except Exception as e:
        print(f"Could not load LoRA: {e}")

    try:
        print("Compiling the model... this may take a minute.")
        dummy_images = [Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))]
        optimize_pipeline_(pipe, image=dummy_images, prompt="a cat")
        print("Compilation complete.")
    except Exception as e:
        print(f"AOT compile failed: {e}")

    return pipe

MAX_SEED = np.iinfo(np.int32).max

def use_output_as_input(output_images):
    """Convert output images to input format for the gallery"""
    if output_images is None or len(output_images) == 0:
        return []
    return output_images

@spaces.GPU(duration=300)
def infer(
    images,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=8,
    height=None,
    width=None,
    rewrite_prompt=True,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    if prompt is None or prompt == "":
        raise gr.Error("Please enter an edit instruction.")
        
    pipe = load_pipeline()
    negative_prompt = " "
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    pil_images = []
    if images is not None:
        for item in images:
            img_obj = item if isinstance(item, Image.Image) else item.get('image', item.get('composite'))
            if isinstance(img_obj, str): 
                pil_images.append(Image.open(img_obj).convert("RGB"))
            elif isinstance(img_obj, Image.Image):
                pil_images.append(img_obj.convert("RGB"))

    if height == 256 and width == 256:
        height, width = None, None

    final_prompt = prompt
    if rewrite_prompt and len(pil_images) > 0:
        print(f"Original User Prompt: '{prompt}'")
        final_prompt = polish_prompt_hf(prompt, pil_images)
        print(f"Rewritten Prompt: '{final_prompt}'")
    
    print(f"Calling pipeline with prompt: '{final_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}, Size: {width}x{height}")
    
    try:
        outputs = pipe(
            image=pil_images if len(pil_images) > 0 else None,
            prompt=final_prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        return outputs, seed, gr.update(visible=True)
    except Exception as e:
        print(f"Inference error: {e}")
        raise e

css = """
#col-container { margin: 0 auto; max-width: 1024px; }
#logo-title { text-align: center; }
#logo-title img { width: 400px; }
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css, title="Qwen-Image Edit Plus") as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <div id="logo-title">
            <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Edit Logo">
            <h2 style="font-style: italic;color: #5b47d1;margin-top: -27px !important;margin-left: 96px">[Plus] Fast, 8-steps with Lightning LoRA</h2>
        </div>
        """)
        gr.Markdown("""
        [Learn more](https://github.com/QwenLM/Qwen-Image) about the Qwen-Image series. 
        This demo uses the new [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) with the [Qwen-Image-Lightning v2](https://huggingface.co/lightx2v/Qwen-Image-Lightning) LoRA + [AoT compilation & FA3](https://huggingface.co/blog/zerogpu-aoti) for accelerated inference.
        Try on [Qwen Chat](https://chat.qwen.ai/), or [download model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) to run locally with ComfyUI or diffusers.
        """)
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input Images", show_label=False, type="pil", interactive=True)
            with gr.Column():
                result = gr.Gallery(label="Result", show_label=False, type="pil")
                use_output_btn = gr.Button("↗️ Use as input", variant="secondary", size="sm", visible=False)

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                placeholder="describe the edit instruction",
                container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                true_guidance_scale = gr.Slider(label="True guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                num_inference_steps = gr.Slider(label="Number of inference steps", minimum=1, maximum=40, step=1, value=8)
            with gr.Row():
                height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024)
                width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024)
            rewrite_prompt = gr.Checkbox(label="Rewrite prompt", value=True)
            num_outputs = gr.Slider(label="Number of Outputs", minimum=1, maximum=4, step=1, value=1)


    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_images, prompt, seed, randomize_seed, true_guidance_scale,
            num_inference_steps, height, width, rewrite_prompt, num_outputs
        ],
        outputs=[result, seed, use_output_btn],
    )

    use_output_btn.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_images]
    )

if __name__ == "__main__":
    demo.launch()