

## Install

1. Clone this repository

```
mkdir LLM_EDQ
cd LLM_EDQ
git clone --recursive https://github.com/xiezx23/llm-edq.git
cd llm-edq
```

2. Install Package

Install Package for CUDA Device
    `CUDA12.6 + torch2.6`

```
conda create -n edq python=3.10 -y
conda activate edq
pip install --upgrade pip
# pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.6.0
pip install --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu126 torch==2.6.0
pip install -e .
```

Install Package for CANN Device (Currently support Ascend 310P)
`CANN8.0.0 + torch2.4 + torch_npu2.4`

```
conda create -n edq python=3.10 -y
conda activate edq
pip install attrs cython numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20 scipy requests absl-py --user
wget https://download.pytorch.org/whl/cpu/torch-2.4.0%2Bcpu-cp310-cp310-linux_x86_64.whl
pip3 install torch-2.4.0+cpu-cp310-cp310-linux_x86_64.whl
wget https://gitee.com/ascend/pytorch/releases/download/v6.0.0-pytorch2.4.0/torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip3 install torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install transformers==4.46.0 accelerate tiktoken einops datasets transformers_stream_generator==0.0.4 peft deepspeed modelscope
```

3. Get the Model and Calibration Dataset

```
python -m utils.download
```

4. Compile CUDA Kernel to Accelrate Infer

```
sh script/compile_kernel.sh
```

If an error occurs:

```
The detected CUDA version (xx.x) mismatches the version that was used to compile PyTorch (12.1). Please make sure to use the same CUDA versions.
```

You can solve by running:

```
conda install -c "nvidia/label/cuda-12.6.0" cuda-toolkit
```



## Usage
On Cloud Server with Multi Device:

```
CUDA_VISIBLE_DEVICES=n python -m edq.main
# n = 2 when use china-mobile server.
# Use python -m edq.main --help for more usage.
```


On CUDA Device:

```
python -m edq.main

# Use python -m edq.main --help for more usage.
```

On CANN Device:

```
python -m edq.ascend_run
```



## Evaluation

Please refer to [eval/README.md](eval/README.md) for evaluation.
The evaluation result is here: [eval/Result.md](eval/Result.md).



## Performance

| 量化方式\指标      | 模型体积↓ （压缩率） | 峰值显存占用↓ ：输入长度8K（显存降低比率） | Wiki2 PPL↓ ：RTN/平滑（性能下降比率） |
| ------------------ | -------------------- | ------------------------------------------ | ------------------------------------- |
| float16 全精度模型 | 14.23 GB             | 15.76 GB                                   | 7.4574                                |
| W4A16 量化         | 5.35 GB（63.25%）    | 6.89 GB（56.28%）                          | 7.8451 / 7.7035（5.20% / 3.30%）      |
| W8A8 量化          | 8.14 GB（42.80%）    | 9.67 GB（38.64%）                          | 7.6172 / 7.5482（2.14% / 1.22%）      |
| EDQ量化：att8 mlp4 | 5.71 GB（59.87%）    | 7.26 GB（53.93%）                          | 7.7263 / 7.6010（3.60% / 1.93%）      |



## Notice to Developers

### Update submodules if you forget to use --recurses when cloning

```
git submodule update --init --recursive
```

### Update your branch to Main

```
git switch main
git pull
git switch your_branch
git merge main
```