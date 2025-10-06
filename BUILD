load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_docker//docker:docker.bzl", "container_image")
load("@pip_deps//:requirements.bzl", "requirement")

filegroup(
    name = "app_srcs",
    srcs = glob([
        "**/*.py",
        "qwenimage/**/*.py",
    ]),
)

py_binary(
    name = "compile_model_bin",
    srcs = ["compile_model.py"],
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        requirement("torchao"),
        requirement("diffusers"),
        requirement("transformers"),
        requirement("sentencepiece"),
        requirement("safetensors"),
        requirement("accelerate"),
        requirement("peft"),
        requirement("huggingface-hub"),
        requirement("kernels"),
        requirement("dashscope"),
        requirement("gradio"),
    ],
)

genrule(
    name = "compiled_model",
    tools = [":compile_model_bin"],
    cmd = "$(location :compile_model_bin)",
    outs = ["compiled_pipe.pt"],
    tags = ["requires-gpu"],
    timeout = "long",
)

container_image(
    name = "qwen_image_server",
    base = "@runpod_pytorch_base//image",
    files = [":app_srcs"],
    data_path = ".",
    data = [":compiled_model"],
    # Copy Python dependencies from our pip_parse rule
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        requirement("torchao"),
        requirement("diffusers"),
        requirement("transformers"),
        requirement("sentencepiece"),
        requirement("safetensors"),
        requirement("accelerate"),
        requirement("peft"),
        requirement("huggingface-hub"),
        requirement("kernels"),
        requirement("dashscope"),
        requirement("gradio"),
        requirement("runpod"),
    ],
    entrypoint = ["/usr/bin/python3", "/app/handler.py"],
)

container_pull(
    name = "runpod_pytorch_base",
    registry = "docker.io",
    repository = "runpod/pytorch",
    tag = "2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
)