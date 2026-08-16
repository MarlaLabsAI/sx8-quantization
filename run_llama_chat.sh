#!/bin/bash
# ============================================================================
# run_llama_chat.sh — S-X8 chat a las VELOCIDADES DECLARADAS (llama.cpp fork)
#
# Decode ~63.79 tok/s · prompt ~1877 tok/s · VRAM ~4-5 GB (Qwen3.5-4B-SX8v43.gguf)
# (los números del paper, medidos en RTX 5060 Ti).
#
# Uso (una vez, para construir el fork):
#   1) ./run_llama_chat.sh build
#   2) ./run_llama_chat.sh           (chat interactivo)
#
# Requisitos: cmake, gcc, CUDA toolkit (nvcc), git.
# ============================================================================
set -e
FORK_DIR="${LLAMA_CPP_DIR:-llama-cpp-sx8}"
GGUF="${GGUF:-Qwen3.5-4B-SX8v43.gguf}"
UPSTREAM_COMMIT="7c203670f"
CUDA_ARCH="${CUDA_ARCH:-120}"   # RTX 50 (Blackwell). Para otras GPUs: 89 (Ada), 86 (Ampere), 80 (A100)

if [ ! -f "$GGUF" ]; then
    echo "ERROR: no encuentro $GGUF. Descárgalo de:"
    echo "  https://huggingface.co/marlalabsAI/Qwen3.5-4B-SX8"
    exit 1
fi

if [ "$1" = "build" ]; then
    echo "==> Construyendo el fork llama.cpp con soporte S-X8..."
    if [ ! -d "$FORK_DIR" ]; then
        git clone https://github.com/ggml-org/llama.cpp "$FORK_DIR" || \
        git clone https://github.com/ggerganov/llama.cpp "$FORK_DIR"
    fi
    cd "$FORK_DIR"
    git checkout "$UPSTREAM_COMMIT" 2>/dev/null || git checkout master
    git apply ../llama-cpp-sx8.patch
    cmake -B build -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    cmake --build build --config Release -j"$(nproc)"
    echo "==> Build completado. Ejecuta de nuevo: ./run_llama_chat.sh"
    exit 0
fi

if [ ! -x "$FORK_DIR/build/bin/llama-cli" ]; then
    echo "El fork no está construido. Ejecuta primero: ./run_llama_chat.sh build"
    exit 1
fi

echo "==> S-X8 chat (llama.cpp fork) — Ctrl+C para salir"
exec "$FORK_DIR/build/bin/llama-cli" \
    -m "$GGUF" \
    -ngl 99 \
    -c 4096 \
    --jinja \
    --temp 0.7 \
    --top-p 0.9 \
    --repeat-penalty 1.1
