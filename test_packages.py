import torch
import torchvision
import pkg_resources  # To list all installed packages

def main():    
    print("📌 **Environment Information:**\n")

    # Torch & CUDA Info
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.backends.mps.is_available():
        print("✅ Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Using CPU (Training will be slow!)")

    print("\n📌 **Installed Packages:**\n")

    # List all installed packages
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    for package, version in sorted(installed_packages.items()):
        print(f"{package}: {version}")

if __name__ == "__main__":
    main()
