import torch
import pytest
import json
import argparse
import os
from common import distributed_test
from simple_model import UnusedParametersModel, random_dataloader, args_from_dict
from deepspeed.ops.op_builder import CPUAdamBuilder

import deepspeed


@pytest.mark.parametrize('find_unused_parameters', [False, True])
def test_stage2_find_unused_parameters(tmpdir, find_unused_parameters):
    pytest.skip('skip for now')
    use_cpu_offload = True

    if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")

    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": 2,
            "cpu_offload": use_cpu_offload,
            "find_unused_parameters": find_unused_parameters
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    }

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 4

    model = UnusedParametersModel(hidden_dim=hidden_dim)

    @distributed_test(world_size=[1])
    def _test_stage2_find_unused_parameters(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                                  model=model,
                                                  model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device)

        def _loop():
            for n, batch in enumerate(data_loader):
                loss = model(batch[0], batch[1])
                model.backward(loss)
                model.step()

        if not find_unused_parameters:
            with pytest.raises(AssertionError) as e:
                _loop()
            assert e.value.args and 'find_unused_parameters' in e.value.args[0]
        else:
            _loop()

    _test_stage2_find_unused_parameters(args=args, model=model, hidden_dim=hidden_dim)
