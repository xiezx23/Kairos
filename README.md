# Kairos
Kairos is a framework to accerelate quantized LLM inference.
Our key idea is to decouple storage and computation.
For storage, a 4-bit weight quantization is used to reduce memory overhead. 
For computation, different quantization strategies are used to accelerate prefilling at runtime.

![Overview of Kairos](images/Kairos_overview.jpg) 
We decouple the storage of weights of linear layers and the computation of linear layers.
Kairos consists of quantizer, profiler, and converter.
The quantizer compresses a full precision LLM into a 4-bit LLM in Section~\ref{subsec:TSQ}.
The profiler explores the quantization strategy based on Theorem~\ref{theorem} to construct a lookup table for each linear layer in section~\ref{subsec:profiler}.
The converter converts the weights and activations based on the given quantization strategy in Section~\ref{subsec:converter}.

## Install
1. Install Package

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

3. Get the Model and Calibration Dataset

```
python -m utils.download
```

4. Compile CUDA Kernels

```
sh script/compile_kernel.sh
```

## Usage

On CUDA Device:

```
python -m edq.main

# Use python -m edq.main --help for more usage.
```

## Evaluation

Please refer to [eval/README.md](eval/README.md) for evaluation.
The evaluation result is here: [eval/Result.md](eval/Result.md).

## Performance
Kairos provides 1.57$\times$, and 1.63$\times$ speedups on average for LLaMa-3 8B, and 1.47$\times$, and 1.61$\times$ speedups on average for Qwen2.5 7B A100 GPU, compared with SOTA W4A16 method MARLIN and W4A8 method Qserve, respectively.
We also evaluate the performance of Kairos on Nvidia Jetson AGX Orin.
When the input dimension is set to 64K, 
Kairos provides up to 1.27$\times$, 1.42$\times$ and 1.40$\times$ speedups for LLaMa-3 8B, 
and up to 1.23$\times$, 1.38$\times$ and 1.54$\times$ speedups for Qwen2.5 7B, 
compared with MARLIN, AWQ and Qserve, respectively, in Figure~\ref{fig:speedups}(c) and (d).
The overall performance of Kairos is significantly higher than other quantization methods.
This is because Kairos decouples storage and computation for dynamic workloads in LLM prefilling.
In addition, the performance of Kairos is optimal for some specific workloads.
This is because the optimal strategies are different among the linear layers of different weight shapes.
![Speedup over cuBLAS of LLM linear layers](images/speedup.png) 
![Total Latency of LLM linear](images/end2endResult.png) 
