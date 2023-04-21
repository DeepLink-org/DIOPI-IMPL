# Copyright (c) 2023, DeepLink.

from .device_config_helper import Skip
from .dtype import Dtype

device_configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)]
                }
            ]
        )
    ),

    'baddbmm': dict(
        name=["baddbmm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)]
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)]
                },
            ]
        ),
    ),

    'hardswish': dict(
        name=["hardswish"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)]
                },
            ]
        ),
    ),

    'avg_pool2d': dict(
        name=["avg_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)]
                }
            ]
        )
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)]
                }
            ]
        )
    ),

    'adaptive_max_pool2d': dict(
        name=["adaptive_max_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)]
                }
            ]
        )
    ),

    'binary_cross_entropy': dict(
        name=["binary_cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)]
                }
            ]
        )
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float32)]
                }
            ]
        )
    ),

    'erfinv': dict(
        name=["erfinv"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)]
                }
            ]
        )
    ),

    'sign': dict(
        name=['sign'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16), Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)]
                }
            ]
        )
    ),

    'sign_zero': dict(
        name=['sign_zero'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)]
                }
            ]
        )
    ),
    
    'silu': dict(
        name=['silu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)]
                }
            ]
        )
    ),
    
    'bmm': dict(
        name=['bmm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)]
                }
            ]
        )
    ),
    
    'matmul': dict(
        name=['matmul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)]
                }
            ]
        )
    ),
    
    'clamp_tensor': dict(
        name=['clamp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)]
                }
            ]
        )
    ),

    'fill': dict(
        name=["fill_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)]
                }
            ]
        )
    ),
    
    'fill_not_float': dict(
        name=["fill_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.float16), Skip(Dtype.int8),
                              Skip(Dtype.uint8), Skip(Dtype.bool)]
                }
            ]
        )
    ),
    
    'reduce_op': dict(
        name=['mean', 'sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float64), Skip(Dtype.float16)],
                },
            ],
        ),
    ),
    
    'reduce_partial_op': dict(
        name=['mean', 'sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float64), Skip(Dtype.float16)],
                },
            ],
        ),
    ),
}
