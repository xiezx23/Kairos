# This file is modified from https://github.com/mit-han-lab/omniserve/omniserve/modeling/layers/quantized_linear/w4a8_linear.py
# File modifier: Zexi Xie
# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }
import omniserve_backend.qgemm_w4a8_per_chn
import omniserve_backend.qgemm_w4a8_per_group
import kairos_cuda_accel
import awq_backend
from thirdparty.Qserve.quantization import *
import torch

def pack_int_qserve(unpacked_qweight):
    N, K = unpacked_qweight.shape
    W_unpack_reorder = (
        unpacked_qweight.reshape(N // 32, 2,2, 8, K // 32, 2, 4, 4,)
        .permute(0, 4, 3, 6, 1, 5, 2, 7)
        .contiguous())
    W_unpack_reorder = (
        W_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7, 4)
        .contiguous()
        # .to(torch.int8)
    )
    return W_unpack_reorder.reshape(N,K)


class W4A8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,

        bias: bool = True,
        group_size: int = 128,
        device: torch.device = torch.cuda.current_device(),
    ):
        super().__init__()

        w_bit = 4
        self.interleave = 1  # Currently no interleave
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.per_channel = group_size == -1
        self.group_size = group_size if group_size != -1 else in_features

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        int8_pack_num = 8 // self.w_bit

        assert out_features % (self.interleave) == 0
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    out_features // self.interleave,
                    in_features // int8_pack_num * self.interleave,
                ),
                dtype=torch.int8,
                device=device,
            ).contiguous(),
        )

        self.register_buffer(
            "s1_scales",
            torch.zeros(
                (out_features,),
                dtype=torch.float16,
                device=device,
            ).contiguous(),
        )

        if self.per_channel:
            # per-channel quantization
            self.register_buffer(
                "s1_szeros",
                torch.zeros(
                    (out_features,),
                    dtype=torch.float16,
                    device=device,
                ).contiguous(),
                # NOTE: this is the scaled zeros
            )
            self.forward = self.forward_per_chn
        else:
            # per-group quantization
            self.register_buffer(
                "s2_scales",
                torch.zeros(
                    (
                        in_features // self.group_size,
                        out_features,
                    ),
                    dtype=torch.int8,
                    device=device,
                ).contiguous(),
            )
            self.register_buffer(
                "s2_zeros",
                torch.zeros(
                    (
                        in_features // self.group_size,
                        out_features,
                    ),
                    dtype=torch.int8,  # Actually 2's complement for sint8 zeros
                    device=device,
                ).contiguous(),
                # NOTE: this is the scaled zeros
            )
            # self.forward = self.forward_per_group

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=device)
            )
        else:
            self.bias = None

    @torch.no_grad()
    def forward_per_chn(self, x, input_scales, input_sum, output_buffer):
        omniserve_backend.qgemm_w4a8_per_chn.gemm_forward_cuda(
            x,
            self.qweight,
            self.s1_scales,
            input_scales,
            self.s1_szeros,
            input_sum,
            output_buffer,
        )
        output_bias = self.bias
        if output_bias is not None:
            output_buffer += output_bias

    @torch.no_grad()
    def forward_per_group(self, x, input_scales, output_buffer):
        # input sum is of no use here. Only to keep the interface consistent
        omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
            x,
            self.qweight,
            self.s2_zeros,
            self.s2_scales,
            self.s1_scales,
            input_scales,
            output_buffer,
        )
        output_bias = self.bias
        if output_bias is not None:
            output_buffer += output_bias

    @torch.no_grad()
    def forward(self, input):
        intput_shape = input.shape
        input = input.view(-1, intput_shape[-1])

        input_int8 = torch.empty_like(input, device='cuda', dtype=torch.int8)
        scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
        awq_backend.invoke_quant(input_int8, input, scale_x)
        output_buffer = torch.empty((input.shape[0], self.out_features), dtype=torch.float16, device='cuda')

        self.forward_per_group(input_int8, scale_x, output_buffer)
        return output_buffer

    @classmethod
    def from_module(cls, linear, device, return_dtype = torch.float16, init_only=False, name = ''):
        weight_tensor = linear.weight
        bias          = linear.bias
        group_size    = 128
        N, K = weight_tensor.shape
        q_linear = cls(K, N, bias is not None, group_size, device)
        
        if bias is not None:
            q_linear.bias = bias.clone().half()

        weight_int8, qconfig_int8 = qserve_quantize_tensor_int8(weight_tensor)
        weight_int4, qconfig_int4 = qserve_quantize_tensor_int4(weight_int8, group_size=group_size)
        s1_scale = qconfig_int8['scale'].contiguous()
        s2_scale = qconfig_int4['scale'].contiguous()
        s2_zero = qconfig_int4['zero_pt'].contiguous()

        # Step2: Reshape tensor
        linear_weight = weight_int4.reshape(N, K // group_size, group_size)
        s2_zero = s2_zero.reshape(N, K // group_size, 1)
        s2_scale = s2_scale.reshape(N, K // group_size, 1)

        # Step 3: Pack the fake quantized weights to real quantized weights
        # ---- Repack the weight ---- #
        # pack to M // 32, K // 32, (8, 4), ([2], 2, 2, 4)
        W_unpack_reorder = (
            linear_weight.reshape(N // 32, 2,2, 8, K // 32, 2, 4, 4,)
            .permute(0, 4, 3, 6, 1, 5, 2, 7)
            .contiguous())
        W_unpack_reorder = (
            W_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7, 4)
            .contiguous()
            .to(torch.int8)
        )
        # B_fp16_reorder = B_fp16_reorder[:, :, :, :, :, :, [3, 2, 1, 0]].contiguous()
        # [16, 0, 17, 1, ...]
        W_unpack_repacked = (W_unpack_reorder[..., 1] << 4) + W_unpack_reorder[
            ..., 0
        ]
        W_unpack_repacked = W_unpack_repacked.reshape(
            N // 32, K // 32, 32, 16
        ).reshape(N, K // 2)
        q_linear.qweight.data[:, :] = W_unpack_repacked

        # ---- Pack the scales ---- #
        q_linear.s1_scales.data[:] = s1_scale.reshape(N)

        s2_scale = (
            s2_scale.reshape(N, K // group_size)
            .transpose(0, 1)
            .contiguous()
        )
        s2_scale = s2_scale.reshape(
            K // group_size, N // 32, 32)
        s2_scale = (
            s2_scale.reshape(
                K // group_size, N // 32, 4, 8)
            .transpose(-2, -1)
            .contiguous()
        )
        s2_scale = s2_scale.reshape(
            K // group_size, N
        ).contiguous()
        q_linear.s2_scales.data[:, :] = s2_scale

        # ---- Pack the zeros ---- #
        s2_zero = -s2_zero
        s2_zero = s2_zero.int()  # convert to 2-complement

        s2_zero = (
            s2_zero.reshape(N, K // group_size)
            .transpose(0, 1)
            .contiguous()
        )
        s2_zero = s2_zero.reshape(
            K // group_size, N // 32, 32
        )
        # for the last dimension, organize as 0, 8, 16, 24, 1, 9, 17, 25, ... following the requirement of tensor core gemm
        s2_zero = (
            s2_zero.reshape(
                K // group_size, N // 32, 4, 8
            )
            .transpose(-2, -1)
            .contiguous())
        s2_zero = (
            s2_zero.reshape(
                K // group_size, N
            ).contiguous()
            * s2_scale)
        q_linear.s2_zeros.data[:, :] = s2_zero
        return q_linear
    
    @classmethod
    def from_weight(cls, weight_tensor, bias, group_size):
        N, K = weight_tensor.shape
        q_linear = cls(K, N, bias is not None, group_size, 'cuda')

        if bias is not None:
            q_linear.bias = bias.clone().half()

        weight_int8, qconfig_int8 = qserve_quantize_tensor_int8(weight_tensor)
        weight_int4, qconfig_int4 = qserve_quantize_tensor_int4(weight_int8, group_size=group_size)
        s1_scale = qconfig_int8['scale'].contiguous()
        s2_scale = qconfig_int4['scale'].contiguous()
        s2_zero = qconfig_int4['zero_pt'].contiguous()

        # Step2: Reshape tensor
        linear_weight = weight_int4.reshape(N, K // group_size, group_size)
        s2_zero = s2_zero.reshape(N, K // group_size, 1)
        s2_scale = s2_scale.reshape(N, K // group_size, 1)

        # Step 3: Pack the fake quantized weights to real quantized weights
        # ---- Repack the weight ---- #
        # pack to M // 32, K // 32, (8, 4), ([2], 2, 2, 4)
        W_unpack_reorder = (
            linear_weight.reshape(N // 32, 2,2, 8, K // 32, 2, 4, 4,)
            .permute(0, 4, 3, 6, 1, 5, 2, 7)
            .contiguous())
        W_unpack_reorder = (
            W_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7, 4)
            .contiguous()
            .to(torch.int8)
        )
        # B_fp16_reorder = B_fp16_reorder[:, :, :, :, :, :, [3, 2, 1, 0]].contiguous()
        # [16, 0, 17, 1, ...]
        W_unpack_repacked = (W_unpack_reorder[..., 1] << 4) + W_unpack_reorder[
            ..., 0
        ]
        W_unpack_repacked = W_unpack_repacked.reshape(
            N // 32, K // 32, 32, 16
        ).reshape(N, K // 2)
        q_linear.qweight.data[:, :] = W_unpack_repacked

        # ---- Pack the scales ---- #
        q_linear.s1_scales.data[:] = s1_scale.reshape(N)

        s2_scale = (
            s2_scale.reshape(N, K // group_size)
            .transpose(0, 1)
            .contiguous()
        )
        s2_scale = s2_scale.reshape(
            K // group_size, N // 32, 32)
        s2_scale = (
            s2_scale.reshape(
                K // group_size, N // 32, 4, 8)
            .transpose(-2, -1)
            .contiguous()
        )
        s2_scale = s2_scale.reshape(
            K // group_size, N
        ).contiguous()
        q_linear.s2_scales.data[:, :] = s2_scale

        # ---- Pack the zeros ---- #
        s2_zero = -s2_zero
        s2_zero = s2_zero.int()  # convert to 2-complement

        s2_zero = (
            s2_zero.reshape(N, K // group_size)
            .transpose(0, 1)
            .contiguous()
        )
        s2_zero = s2_zero.reshape(
            K // group_size, N // 32, 32
        )
        # for the last dimension, organize as 0, 8, 16, 24, 1, 9, 17, 25, ... following the requirement of tensor core gemm
        s2_zero = (
            s2_zero.reshape(
                K // group_size, N // 32, 4, 8
            )
            .transpose(-2, -1)
            .contiguous())
        s2_zero = (
            s2_zero.reshape(
                K // group_size, N
            ).contiguous()
            * s2_scale)
        q_linear.s2_zeros.data[:, :] = s2_zero
        return q_linear

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
