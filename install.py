import subprocess
import sys

def run(command, description):
    print(f"\n {description}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error occurred while running: {command}")
        sys.exit(1)

# Step 1: Install base dependencies
run("pip install -r requirements.txt", "Installing base dependencies")

# Step 2: Install specific version of PyTorch with CUDA 12.1
run("pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", "Installing PyTorch with CUDA 12.1")

# Step 3: Install MeloTTS from GitHub (skip problematic dependencies)
run("pip install git+https://github.com/myshell-ai/MeloTTS.git@5b538481e24e0d578955be32a95d88fcbde26dc8 --no-deps", "Installing MeloTTS from GitHub (without dependencies)")

# Step 4: Download dictionary for MeCab (used by MeloTTS)
run("python -m unidic download", "Downloading UniDic dictionary for MeCab")

print("\n All packages have been successfully installed!")
