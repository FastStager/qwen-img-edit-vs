load("@rules_python//python:defs.zl", "py_library")

py_library(
    name = "safetensors",
    srcs = glob(["**/*.py"]),
    visibility = ["//visibility:public"],
)