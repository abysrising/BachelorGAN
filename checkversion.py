import torch


def check_version():
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")

def check_cuda_version():
    cuda_version = torch.version.cuda
    print(f"PyTorch CUDA version: {cuda_version}")

if __name__ == "__main__":
    check_version()
    check_cuda_version()
