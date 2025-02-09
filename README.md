# mnist_onnxruntime_cpp
Simple Example of Running MNIST using ONNX Runtime in C++



# Setup

tested on x86_64 Linux with NVIDIA GPU

run `setup.sh` to build opencv into `lib` directory

download and extract `onnxruntime-linux-x64-gpu-1.20.1.tgz` into `lib` directory, source: https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1

use `build.sh`, `run.sh`, `clean_build.sh`, `build_run.sh` to build and run

download `mnist-12.onnx` into `models` directory, source https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
