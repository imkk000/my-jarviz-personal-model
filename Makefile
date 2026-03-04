.PHONY:

CONDA_ENV = unsloth_env
PYTHON = $(HOME)/.conda/envs/$(CONDA_ENV)/bin/python
LLAMA_PATH = ./llama.cpp
MODEL_NAME = jarviz

convert_gguf:
	$(PYTHON) $(LLAMA_PATH)/convert_hf_to_gguf.py $(MODEL_NAME) \
    --outfile $(MODEL_NAME)_bf16.gguf --outtype bf16
	$(LLAMA_PATH)/llama-quantize $(MODEl_NAME)_f16.gguf $(MODEL_NAME)_q4km.gguf q4_k_m

warm_up:
	# unsloth warm up for generating cache
	$(PYTHON) -c "from unsloth import FastLanguageModel"

build_llama.cpp:
	# ensure to install cuda
	git clone https://github.com/ggml-org/llama.cpp
	cmake llama.cpp -B llama.cpp/build \
			-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
	cmake --build llama.cpp/build --config Release -j --clean-first --target llama-vision llama-cli llama-mtmd-cli llama-server llama-gguf-split
	cp llama.cpp/build/bin/llama-* llama.cpp
