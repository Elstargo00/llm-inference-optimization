# LLM Inference Optimization - Reproduction Guide

### Step 0: Clone the project repo

```
git clone git@github.com:Elstargo00/llm-inference-optimization.git

cd llm-inference-optimization
```

### Step 1: Install llama.cpp with GPU Support

Clone and build llama.cpp with CUDA acceleration enabled:

```
# Clone the repository
git clone git@github.com:ggml-org/llama.cpp.git

cd llama.cpp

# Build with CUDA support (choose based on your system)

# For Linux/MacOS:
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j16

# For Windows (using CMake):
mkdir build
cd build
cmake .. -DLLAMA_CUDA=ON
cmake --build . --config Release

# Verify installation
./build/bin/llama-server --help  # Should display help without errors

```

Different OS, build differently. See the offical guide on llama.cpp github repo

### Step2: Download Model Files (.gguf)

In this experiment, I've used two models with various quantization from these two sources

- llava:13b: https://huggingface.co/PsiPi/liuhaotian_llava-v1.5-13b-GGUF
- qwen2.5-vl: https://huggingface.co/Mungert/Qwen2.5-VL-32B-Instruct-GGUF

The models includes:

1. llava-v1.5-13b-f16.gguf + mmproj-model-f16.gguf
2. llava-v1.5-13b-Q8_0.gguf + mmproj-model-Q8_0.gguf
3. llava-v1.5-13b-Q5_K_M.gguf + mmproj-model-Q5_0.gguf
4. Qwen2.5-VL-32B-Instruct-q4_k_m.gguf + Qwen2.5-VL-32B-Instruct-mmproj-f16.gguf
5. Qwen2.5-VL-32B-Instruct-q3_k_m.gguf + Qwen2.5-VL-32B-Instruct-mmproj-f16.gguf

**NOTE:** mmproj is a vision projection paring with the model.

After finish downloading all these. Put it in the `llama.cpp/models` directory

### Step3: Copy & Install Python Dependencies & Configure OpenAI API Key

```

# Move PWD to project root (/llm-inference-optimization)

### STEP1: COPY .env
cp .env.example .env
# then edit .env
# OPENAI_API_KEY='your-api-key-here'

### STEP2: COPY LLAMA.CPP SCRIPTS
cp llamacpp_scripts/*.sh llama.cpp/

### STEP3: MOVE `Llava` and `Qwen` to llama.cpp/models
### MAKE SURE ALL MODEL DOWNLOAD *.gguf file is in llama.cpp/models

### STEP4: CREATE PYTHON VENV & BEGIN OUR EXPERIMENT
uv venv .venv
uv run jupyter lab
```

### Step4: Run the bechmark suite cell

`optimization_experiment.ipynb` is our working directory.
You can test it via Jupyter Lab while running llama.cpp model on another terminal

```
cd llama.cpp

# run model script by
./start_llava_baseline.sh # or other script you want
```

```
from benchmark import VLMBenchmark

# Initialize benchmark
benchmark = VLMBenchmark()


test_cases = [
    ... # you change task name, testing document and adjust prompt
]

# Define test configurations
model_configs = [
    ... # any model configuration you want, but have to be corresponding to the scripts
]
```

![alt text](next_model.png "Change the running model when this input pop up. Then press enter when the next model script is running")

Follow this process:

1. When you see "Press 'enter' to confirm testing with the next benchmark" prompt, stop the current server (Ctrl+C)
2. Navigate to llamacpp_scripts/ directory
3. Run the corresponding configuration script:

```

# Example: running llava Optimize B correspondingly

cd llamacpp_scripts
./start_llava_optimize_B.sh

```

4. Wait for "server is listening on http://0.0.0.0:10000"
5. Press Enter in the notebook to continue

When complete all testsuits, the results will be saved at `benchmark_results` directory
