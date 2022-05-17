torch-model-archiver --model-name mnist --version 1.0 --model-file mnist_model.py --serialized-file mnist_sd.pt --export-path ./model_store --handler mnist_handler.py

torchserve --start --ncs --model-store model_store --models mnist.mar

