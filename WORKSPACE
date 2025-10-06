load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_docker",
    sha256 = "6b2da98a87a206c5598858a2a4816c34e8034b7f7300cf6b694b8e3a24b07d57",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz",
        "https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz",
    ],
)

load("@rules_docker//docker:docker.bzl", "docker_repositories", "container_pull")
docker_repositories()

load("@rules_docker//toolchains/docker:toolchain.bzl", "docker_toolchain_configure")
docker_toolchain_configure(name = "docker_config")

container_pull(
    name = "runpod_pytorch_base",
    registry = "docker.io",
    repository = "runpod/pytorch",
    tag = "2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04",
)


# Rules for managing Python dependencies
http_archive(
    name = "rules_python",
    sha256 = "9d0409541a49646b866847b744b20d73899a803c1b26f54c5e3d2568b2089858",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.27.1/rules_python-0.27.1.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    requirements_lock = "//:requirements.txt",
)

load("@pip_deps//:requirements.bzl", "install_deps")
install_deps()