load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "packaging",
    srcs = glob(["**/*.py"]),
    visibility = ["//visibility:public"],
)