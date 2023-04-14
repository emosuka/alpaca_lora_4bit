# NOT AN INSTALLER SCRIPT, just something I used to spin up a server instance

apt update
apt install nano
apt upgrade python3
apt install python3.10-dev
export CPATH=/usr/include/python3.10/
pip install accelerate
pip install datasets
pip install safetensors
pip install wandb
# pip install git+https://github.com/huggingface/peft.git@70af02a2bca5a63921790036b2c9430edf4037e2
pip install git+https://github.com/sterlind/peft.git
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/sterlind/GPTQ-for-LLaMa.git@lora_4bit
pip install bitsandbytes
pip install colorama
pip install xformers
pip install sentencepiece
