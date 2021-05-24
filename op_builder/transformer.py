"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import torch
from .builder import CUDAOpBuilder, is_rocm_pytorch


class TransformerBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    NAME = "transformer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/transformer/ds_transformer_cuda.cpp',
            'csrc/transformer/cublas_wrappers.cu',
            'csrc/transformer/transform_kernels.cu',
            'csrc/transformer/gelu_kernels.cu',
            'csrc/transformer/dropout_kernels.cu',
            'csrc/transformer/normalize_kernels.cu',
            'csrc/transformer/softmax_kernels.cu',
            'csrc/transformer/general_kernels.cu'
        ]

    def include_paths(self):
        includes = ['csrc/includes']
        if is_rocm_pytorch:
            from torch.utils.cpp_extension import ROCM_HOME
            includes += ['{}/hiprand/include'.format(ROCM_HOME), '{}/rocrand/include'.format(ROCM_HOME)]
        return includes

    def nvcc_args(self):
        args = [
            '-O3',
            '-std=c++14',
        ]
        if is_rocm_pytorch:
            args += [
                '-U__HIP_NO_HALF_OPERATORS__',
                '-U__HIP_NO_HALF_CONVERSIONS__',
                '-U__HIP_NO_HALF2_OPERATORS__'
            ]
        else:
            args += [
                '--use_fast_math',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__'
            ]
            args += self.compute_capability_args()
        return args
