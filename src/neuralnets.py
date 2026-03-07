from typing import Optional, List, Union
import hashlib
import json

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.auto import (AutoGRU,
                                 AutoNBEATS,
                                 AutoTiDE,
                                 AutoNLinear,
                                 AutoKAN,
                                 AutoMLP,
                                 AutoLSTM,
                                 AutoDLinear,
                                 AutoNHITS,
                                 AutoPatchTST,
                                 AutoTFT,
                                 AutoDeepNPTS,
                                 AutoDeepAR,
                                 AutoTCN,
                                 AutoDilatedRNN)
from ray.tune.search.variant_generator import generate_variants

from neuralforecast.models import (GRU,
                                   KAN,
                                   NBEATS,
                                   TiDE,
                                   NLinear,
                                   MLP,
                                   LSTM,
                                   DLinear,
                                   NHITS, DeepAR,
                                   PatchTST,
                                   TFT,
                                   DeepNPTS,
                                   DeepAR,
                                   TCN,
                                   DilatedRNN)


class BaseModelsConfig:
    AUTO_MODEL_CLASSES = {
        'AutoTFT': AutoTFT,
        'AutoNBEATS': AutoNBEATS,
        # 'AutoTiDE': AutoTiDE,
        # 'AutoNLinear': AutoNLinear,
        # 'AutoKAN': AutoKAN,
        # 'AutoMLP': AutoMLP,
        # 'AutoDLinear': AutoDLinear,
        # 'AutoNHITS': AutoNHITS,
        # 'AutoDeepNPTS': AutoDeepNPTS,
        # 'AutoPatchTST': AutoPatchTST,

        # 'AutoGRU': AutoGRU,
        # 'AutoDeepAR': AutoDeepAR,
        # 'AutoLSTM': AutoLSTM,
        # 'AutoDilatedRNN': AutoDilatedRNN,
        # 'AutoTCN': AutoTCN,
    }

    MODEL_CLASSES = {
        'AutoKAN': KAN,
        'AutoMLP': MLP,
        'AutoDLinear': DLinear,
        'AutoNHITS': NHITS,
        'AutoDeepNPTS': DeepNPTS,
        'AutoNBEATS': NBEATS,
        'AutoTiDE': TiDE,
        'AutoNLinear': NLinear,
        'AutoTFT': TFT,
        'AutoPatchTST': PatchTST,
        'AutoGRU': GRU,
        'AutoDeepAR': DeepAR,
        'AutoLSTM': LSTM,
        'AutoDilatedRNN': DilatedRNN,
        'AutoTCN': TCN,
    }

    NEED_CPU = ['AutoGRU',
                'AutoDeepNPTS',
                # 'AutoTFT',
                'AutoPatchTST',
                'AutoDeepAR',
                'AutoLSTM',
                'AutoTiDE',
                'AutoNLinear',
                'AutoKAN',
                'AutoDilatedRNN',
                'AutoTCN']

    @classmethod
    def get_pseudo_auto_nf_models(cls,
                                  horizon: int,
                                  input_size: int,
                                  n_samples: int,
                                  try_mps: bool = True,
                                  limit_epochs: bool = False,
                                  limit_val_batches: Optional[int] = None):
        """

        :param horizon:
        :param input_size:
        :param n_samples:
        :param try_mps:
        :param limit_epochs:
        :param limit_val_batches:
        :return:

        example:

        BaseModelsConfig.get_pseudo_auto_nf_models(horizon=12,
                                           input_size=12,
                                           n_samples=10,
                                           try_mps=True)


        """

        models = []
        for mod_name, mod in cls.AUTO_MODEL_CLASSES.items():

            if try_mps:
                if mod_name in cls.NEED_CPU:
                    mod.default_config['accelerator'] = 'cpu'
                else:
                    mod.default_config['accelerator'] = 'mps'
            else:
                mod.default_config['accelerator'] = 'cpu'

            if limit_epochs:
                mod.default_config['max_steps'] = 2

            if limit_val_batches is not None:
                mod.default_config['limit_val_batches'] = limit_val_batches

            configs = cls.sample_configs(model_name='',
                                         config=mod.default_config,
                                         horizon=horizon,
                                         n_samples=n_samples)

            for i, conf_ in enumerate(configs):
                conf_['h'] = horizon
                conf_.pop('loss')
                if 'input_size' not in conf_.keys():
                    conf_['input_size'] = input_size

                mod_name_ = cls.MODEL_CLASSES.get(mod_name).__name__

                model_inst = cls.MODEL_CLASSES.get(mod_name)(
                    **conf_,
                    alias=f'{mod_name_}_{i}'
                )

                # model_instance = mod(
                #     h=horizon,
                #     num_samples=n_samples,
                #     alias=mod_name,
                #     valid_loss=MAE(),
                #     refit_with_val=True,
                # )

                models.append(model_inst)

        return models

    @classmethod
    def sample_configs(cls, model_name: str, horizon: int, n_samples: int = 10, config: Optional[None] = None):
        # BaseModelsConfig.sample_configs('AutoTFT', horizon=2, n_samples=5)

        if config is None:
            config_pool = cls.AUTO_MODEL_CLASSES.get(model_name).get_default_config(h=horizon, backend='ray')
        else:
            config_pool = config

        samples = []
        # For a random search space, generate_variants yields once per call; use different seeds
        for seed in range(n_samples):
            gen = generate_variants({"config": config_pool}, random_state=seed)
            resolved_vars, spec = next(gen)
            samples.append(spec["config"])

        return samples
