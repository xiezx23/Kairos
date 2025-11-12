import torch

# import edq_cuda_accel
import awq_backend
import kernels.edq_cuda_accel as edq_cuda_accel

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

M, N, K = 4096, 18944, 3584 # gate_proj, up_proj
# M, N, K = 1024, 3584, 18944 # down_proj
# M, N, K = 1024, 3584, 3584 # q_proj, o_proj
# M, N, K = 1024, 512, 3584 # kv_proj

# M = 1024 * 1
# M = 128 * 10
# M = 1024 * 32
# M = 1024 * 64
# M = 2048
M = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype_origin = torch.float16
dtype_q = torch.int8
# dtype_q = torch.uint8

# 指定要测试的函数名称
# test_function_name = "i8_dq"
# test_function_name = "fp16_gemm"
# test_function_name = "awq_w8a8_gemm"

# test_function_name = "w8a16_ws_gemm"
test_function_name = "w8a16_ws_gemv"
# test_function_name = "w8a16_wa_gemm"
# test_function_name = "w8a8_wsas_gemm"
# test_function_name = "w8a8_wsas_gemv"
verbose = False

# 均匀分布
# A = torch.rand(M, K, device=device, dtype=dtype_origin) * 2 - 1    # Matrix A with shape (M, K)
# B = torch.rand(N, K, device=device, dtype=dtype_origin) * 2 - 1    # Matrix B with shape (N, K)

# 标准分布
A = torch.randn(M, K, device=device, dtype=dtype_origin)    # Matrix A with shape (M, K)
B = torch.randn(N, K, device=device, dtype=dtype_origin)    # Matrix B with shape (N, K)


# 非对称量化
def quantize_asym(x, bit = 8):
    x_shape = x.shape
    x = x.to(torch.float32)
    # 对第一个维度(m)进行量化，每行独立计算scale和zp
    max_val = x.amax(dim=-1, keepdim=True)  # shape: (m, 1)
    min_val = x.amin(dim=-1, keepdim=True)  # shape: (m, 1)
    
    if dtype_q == torch.int8:
        max_int = 2**(bit - 1) - 1
        scale = (max_val - min_val).clamp(min=1e-7) / 2 / max_int  # shape: (m, 1)
        zp = - torch.round(input=(max_val + min_val) / 2 / scale)  # shape: (m, 1)
        x.div_(scale).round_().add_(zp)
        x.clamp_(min=-max_int, max=max_int)

    elif dtype_q == torch.uint8:
        max_int = 2**bit - 1
        scale = (max_val - min_val).clamp(min=1e-7) / max_int  # shape: (m, 1)
        zp = - torch.round(input=min_val / scale)  # shape: (m, 1)
        x.div_(scale).round_().add_(zp)
        x.clamp_(min=0, max=max_int)
    
    # 将scale和zp压缩为1维
    scale = scale.squeeze(-1).to(torch.float16)  # shape: (m,)
    zp = zp.squeeze(-1).to(torch.float16)        # shape: (m,)
    
    return x.reshape(x_shape).to(dtype_q), scale, zp

# 对称量化
def quantize_sym(x, bit = 8):
    x_shape = x.shape
    # 将x展为2维
    x = x.reshape(-1, x_shape[-1])
    x = x.to(torch.float32)
    max_int = 2**(bit - 1) - 1

    max_val = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
    scale = max_val / max_int
    x.div_(scale).round_()
    x.clamp_(min=-max_int, max=max_int)
    
    scale = scale.squeeze(-1).to(torch.float16)
    zp = torch.zeros(scale.shape, device=device, dtype=torch.float16)

    return x.reshape(x_shape).to(dtype_q), scale, zp

quantize = {
    "asym" : quantize_asym,
    "sym" : quantize_sym
}

def dequantize_asym(x, scale, zp):
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1]).to(torch.float16)
    x.sub_(zp.view(-1, 1)).mul_(scale.view(-1, 1))

    return x.reshape(x_shape)

def dequantize_sym(x, scale, zp=None):
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1]).to(torch.float16)
    x.mul_(scale.view(-1, 1))

    return x.reshape(x_shape)

dequantize = {
    "asym" : dequantize_asym,
    "sym" : dequantize_sym
}

def post_dequantize(x, scale_a, scale_b):
    x_shape = x.shape
    x = x.to(torch.float32).reshape(-1, x_shape[-1])
    x.mul_(scale_a.to(torch.float32).view(-1, 1)).mul_(scale_b.to(torch.float32).view(1, -1))
    return x.to(torch.float16).reshape(x_shape)

A_q = torch.empty_like(A, dtype=torch.int8, device=device)
A_s = torch.empty(A.shape[0], dtype=torch.float16, device=device)
B_sq = torch.empty_like(B, dtype=torch.int8, device=device)
B_ss = torch.empty(B.shape[0], dtype=torch.float16, device=device)

edq_cuda_accel.quant_fp16_to_int8(A_q, A, A_s)
edq_cuda_accel.quant_fp16_to_int8(B_sq, B, B_ss)

# A_q, A_s, A_zp = quantize["sym"](A, bit=8)
# B_sq, B_ss, B_szp = quantize["sym"](B, bit=8)

B_aq, B_as, B_azp = quantize["asym"](B, bit=8)


A_dq = dequantize["sym"](A_q, A_s, None)
B_dq = dequantize["sym"](B_sq, B_ss, None)

def i8_dq(A_q, B_q, A_s, B_s, C):
    C_q = torch._int_mm(input=A_q, mat2=B_q.T)
    C = post_dequantize(C_q, A_s, B_s)
    return

# 创建输出矩阵
C_predef = torch.zeros(size=(M, N), dtype=torch.float16, device=device)

# 定义测试函数配置
test_functions = {
    # W8A16 functions
    "w8a16_ws_gemm": {
        "func": edq_cuda_accel.w8a16_ws_gemm_cuda,
        "args": lambda: (A, B_sq, B_ss, C_predef),
        "description": "W8A16 Weight-Symmetry GEMM"
    },
    "w8a16_ws_gemv": {
        "func": edq_cuda_accel.w8a16_ws_gemv_cuda,
        "args": lambda: (A, B_sq, B_ss, C_predef),
        "description": "W8A16 Weight-Symmetry GEMV"
    },
    "w8a16_wa_gemm": {
        "func": edq_cuda_accel.w8a16_wa_gemm_cuda,
        "args": lambda: (A, B_aq, B_as, B_azp, C_predef),
        "description": "W8A16 Weight-ASymmetry GEMM"
    },
    
    # W8A8 functions
    "w8a8_wsas_gemm": {
        "func": edq_cuda_accel.w8a8_wsas_gemm_cuda,
        "args": lambda: (A_q, B_sq, A_s, B_ss, C_predef),
        "description": "W8A8 Weight-Symmetry Activation-Symmetry GEMM"
    },
    "w8a8_wsas_gemv": {
        "func": edq_cuda_accel.w8a8_wsas_gemv_cuda,
        "args": lambda: (A_q, B_sq, A_s, B_ss, C_predef),
        "description": "W8A8 Weight-Symmetry Activation-Symmetry GEMV"
    },
    "awq_w8a8_gemm": {
        "func": awq_backend.w8a8_gemm_forward_cuda,
        "args": lambda: (A_q, B_sq, B_ss, A_s, C_predef),
        "description": "AWQ W8A8 GEMM"
    },
    
    # I8 torch function
    "i8_dq": {
        "func": i8_dq,
        "args": lambda: (A_q, B_sq, A_s, B_ss, C_predef),
        "description": "INT8 Dequantize GEMM"
    },
    
    # Full precision
    "fp16_gemm": {
        "func": lambda a, b, c: torch.mm(a, b.T, out=c),
        "args": lambda: (A, B, C_predef),
        "description": "FP16 Full Precision GEMM"
    }
}

def calculate_mse_with_fp16(func_name, verbose):
    """
    计算指定函数与全精度FP16 GEMM的MSE误差
    
    Args:
        func_name: 要测试的函数名称
        verbose: 是否输出误差测试结果
    
    Returns:
        mse: 均方误差值
    """
    if func_name not in test_functions:
        print(f"Error: Function '{func_name}' not found.")
        return None
    
    if func_name == "fp16_gemm":
        print("FP16 GEMM is the reference, MSE = 0.0")
        return 0.0
    
    # 创建输出矩阵
    C_test = torch.zeros(size=(M, N), dtype=torch.float16, device=device)
    C_fp16 = torch.zeros(size=(M, N), dtype=torch.float16, device=device)
    
    # 计算全精度结果作为参考
    fp16_func = test_functions["fp16_gemm"]["func"]
    fp16_args = test_functions["fp16_gemm"]["args"]()
    fp16_args = (*fp16_args[:-1], C_fp16)  # 替换输出矩阵
    fp16_func(*fp16_args)
    
    # 计算测试函数结果
    test_config = test_functions[func_name]
    test_func = test_config["func"]
    test_args = test_config["args"]()
    test_args = (*test_args[:-1], C_test)  # 替换输出矩阵
    
    try:
        test_func(*test_args)
        torch.cuda.synchronize()
        
        # 计算MSE
        mse = torch.mean((C_test - C_fp16) ** 2).item()
        print(f"MSE vs FP16 GEMM: {mse:.2e}")

        if verbose:
            print(f"test {func_name} result: ")
            print(C_test)
            print(f"fp16 GEMM result: ")
            print(C_fp16)

        # 计算最大绝对误差
        max_abs_error = torch.max(torch.abs(C_test - C_fp16)).item()
        print(f"Max Absolute Error: {max_abs_error:.2e}")
        
        return mse
        
    except Exception as e:
        print(f"Error calculating MSE: {e}")
        return None

def benchmark_function(func_name, warmup_iters=50, test_iters=1000, calculate_error=True, verbose=False):
    """
    基准测试函数
    
    Args:
        func_name: 要测试的函数名称
        warmup_iters: 预热迭代次数
        test_iters: 测试迭代次数
        calculate_error: 是否计算与FP16的误差
    """
    if func_name not in test_functions:
        available_funcs = list(test_functions.keys())
        print(f"Error: Function '{func_name}' not found. Available functions: {available_funcs}")
        return
    
    test_config = test_functions[func_name]
    func = test_config["func"]
    get_args = test_config["args"]
    description = test_config["description"]
    
    print(f"\nTesting: {description} ({func_name})")
    print("-" * 60)

    # 计算误差（如果需要）
    mse = None
    if calculate_error:
        print("Calculating accuracy vs FP16 GEMM...")
        mse = calculate_mse_with_fp16(func_name, verbose)
        print()
    
    # 预热
    print(f"Warming up for {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        try:
            func(*get_args())
        except Exception as e:
            print(f"Error during warmup: {e}")
            return
    
    torch.cuda.synchronize()
    
    # 基准测试
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    print(f"Running benchmark for {test_iters} iterations...")
    starter.record()
    for _ in range(test_iters):
        try:
            func(*get_args())
        except Exception as e:
            print(f"Error during benchmark: {e}")
            return
    
    ender.record()
    torch.cuda.synchronize()
    
    curr_time = starter.elapsed_time(ender)
    mean_time = curr_time / test_iters
    
    # 输出结果总结
    print(f"\nResults Summary:")
    print(f"  - Average time: {mean_time:.4f}ms")
    print(f"  - Total time ({test_iters} iters): {curr_time:.4f}ms")
    if mse is not None and func_name != "fp16_gemm":
        print(f"  - MSE vs FP16: {mse:.2e}")
    
    return {"mean_time": mean_time, "mse": mse}


def compare_multiple_functions(func_names, warmup_iters=50, test_iters=1000):
    """
    比较多个函数的性能和精度
    
    Args:
        func_names: 要比较的函数名称列表
        warmup_iters: 预热迭代次数
        test_iters: 测试迭代次数
    """
    results = {}
    print("=" * 80)
    print("PERFORMANCE AND ACCURACY COMPARISON")
    print("=" * 80)
    
    # 首先运行FP16作为基准
    if "fp16_gemm" in func_names:
        results["fp16_gemm"] = benchmark_function("fp16_gemm", warmup_iters, test_iters, calculate_error=False)
        func_names = [name for name in func_names if name != "fp16_gemm"]  # 移除以避免重复
    
    # 运行其他函数
    for func_name in func_names:
        if func_name in test_functions:
            results[func_name] = benchmark_function(func_name, warmup_iters, test_iters)
        else:
            print(f"Warning: Function '{func_name}' not found, skipping...")
    
    # 生成对比表格
    print("\n" + "=" * 100)
    print(f"COMPARISON SUMMARY M,N,K={M},{N},{K}")
    print("=" * 100)
    print(f"{'Function':<50} {'Avg Time (ms)':<15} {'MSE vs FP16':<15} {'Speedup':<10}")
    print("-" * 100)
    
    fp16_time = results.get("fp16_gemm", {}).get("mean_time", None)
    
    for func_name, result in results.items():
        if result is None:
            continue
            
        mean_time = result["mean_time"]
        mse = result.get("mse", 0.0 if func_name == "fp16_gemm" else None)
        
        # 计算加速比
        speedup = f"{fp16_time/mean_time:.2f}x" if fp16_time and mean_time else "N/A"
        
        # 格式化MSE
        mse_str = "0.0" if func_name == "fp16_gemm" else f"{mse:.2e}" if mse is not None else "N/A"
        
        print(f"{test_functions[func_name]['description']:<50} {mean_time:<15.4f} {mse_str:<15} {speedup:<10}")
    
    return results

# 执行基准测试
# results = benchmark_function("awq_w8a8_gemm", verbose=verbose)
# results = benchmark_function(test_function_name, verbose=verbose)

# 如果要测试多个函数，取消注释下面的代码：

# test_functions_list = ["fp16_gemm", "awq_w8a8_gemm", "w8a16_ws_gemm", "w8a16_ws_gemv", "w8a16_wa_gemm", "w8a8_wsas_gemm", "w8a8_wsas_gemv"]
# test_functions_list = ["fp16_gemm", "w8a16_ws_gemm", "w8a16_ws_gemv"]
test_functions_list = ["fp16_gemm", "awq_w8a8_gemm", "w8a8_wsas_gemv"]
test_functions_list = ["fp16_gemm", "w8a8_wsas_gemv", "awq_w8a8_gemm"]

compare_results = compare_multiple_functions(test_functions_list)
