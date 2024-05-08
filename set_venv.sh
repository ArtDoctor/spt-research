python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc