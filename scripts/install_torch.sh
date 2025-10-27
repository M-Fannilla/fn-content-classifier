if [ -z "$CUDA_VER_RAW" ]; then
    echo "⚠️ CUDA not detected -> Installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    exit 0
fi

# Extract major and minor version numbers
CUDA_MAJOR=$(echo "$CUDA_VER_RAW" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER_RAW" | cut -d. -f2)

# Create PyTorch CUDA tag, e.g. 12.4 → cu124
TORCH_CUDA="cu${CUDA_MAJOR}${CUDA_MINOR}"

TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
echo "✅ Using PyTorch index: $TORCH_INDEX"

# Verify URL existence (fallback to CPU if unsupported)
if ! curl --output /dev/null --silent --head --fail "$TORCH_INDEX"; then
    echo "⚠️ Unsupported CUDA version ($CUDA_VER_RAW), falling back to CPU build"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    exit 0
fi

echo "✅ Detected CUDA $CUDA_VER_RAW → Installing PyTorch $TORCH_CUDA"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"