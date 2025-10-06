import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from configs.config import cfg

class BaseNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_type: str = "many_to_one_takelast",
        keep_intermediates: bool = False,
        **other_kwargs,
    ):
        self.output_type = output_type
        self.input_channels = input_channels
        self.keep_intermediates = keep_intermediates
        self.padding_lost_per_side = 0
        self.output_stride = 1

        super(BaseNet, self).__init__()
        self.build(**other_kwargs)

    def build(self, **other_kwargs):
        raise NotImplementedError("Метод 'build' должен быть реализован в подклассах.")

    def forward(self, X: torch.Tensor, pad_mask=None) -> torch.Tensor:
        ys = self._forward(X, pad_mask) 
        if self.output_type == "many_to_one_takelast":
            return ys 
        else:
            raise NotImplementedError(f"Неизвестный тип вывода: {self.output_type}")

    def _forward(self, X: torch.Tensor, pad_mask=None) -> torch.Tensor:
        raise NotImplementedError("Метод '_forward' должен быть реализован в подклассах.")

class CustomRNNMixin(object):
    def __init__(self, *args, **kwargs):
        if "batch_first" not in kwargs:
            kwargs["batch_first"] = True
        super().__init__(*args, **kwargs)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output, _ = super().forward(input_tensor) 
        return output 

class CustomGRU(CustomRNNMixin, nn.GRU):
    pass

class CustomLSTM(CustomRNNMixin, nn.LSTM):
    pass

class CGLLayer(nn.Sequential):
    output_size = None

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 5,
        type: str = "cnn",
        stride: int = 1,
        pool: int = None,
        dropout: float = 0.1,
        stride_pos: str = None, 
        batch_norm: bool = True,
        groups: int = 1,
    ):
        layers = []
        self.output_size = output_size
        self.layer_type = type

        if type == "cnn":
            if dropout:
                layers.append(nn.Dropout1d(dropout)) 
            
            s = 1 if pool else stride
            p = int(np.ceil((kernel_size - s) / 2.0))
            layers.append(
                nn.Conv1d(
                    input_size,
                    output_size,
                    stride=s,
                    kernel_size=kernel_size,
                    padding=p,
                    groups=groups,
                )
            )
            layers.append(nn.ReLU())
            if pool:
                p_pool = int(np.ceil((pool - stride) / 2.0))
                layers.append(
                    nn.AvgPool1d(pool, stride=stride, padding=p_pool, count_include_pad=False)
                )
        elif type in ["gru", "lstm"]:
            klass = {"gru": CustomGRU, "lstm": CustomLSTM}[type]
            
            if dropout:
                layers.append(nn.Dropout(dropout)) 

            assert output_size % 2 == 0, "Output size for bidirectional RNN must be even." 
            layers.append(
                klass(
                    input_size=input_size,
                    hidden_size=int(output_size / 2),
                    bidirectional=True,
                )
            )
        else:
            raise ValueError(f"Неизвестный тип слоя: {type}")

        if batch_norm:
            layers.append(nn.BatchNorm1d(self.output_size))

        super().__init__(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "cnn":
            return super().forward(x)
        elif self.layer_type in ["gru", "lstm"]:
            x_rnn_input = x.transpose(1, 2).contiguous() 
            rnn_output = super().forward(x_rnn_input)    
            return rnn_output.transpose(1, 2).contiguous() 
        else:
            return super().forward(x)

class FilterNetFeatureExtractor(BaseNet):
    def build(
        self,
        n_pre: int = cfg.MODEL_CONFIG['n_pre'],
        w_pre: int = cfg.MODEL_CONFIG['w_pre'],
        n_strided: int = cfg.MODEL_CONFIG['n_strided'],
        w_strided: int = cfg.MODEL_CONFIG['w_strided'],
        n_l: int = cfg.MODEL_CONFIG['n_l'],
        w_l: int = cfg.MODEL_CONFIG['w_l'],
        n_dense_post_l: int = cfg.MODEL_CONFIG['n_dense_post_l'],
        w_dense_post_l: int = cfg.MODEL_CONFIG['w_dense_post_l'],
        cnn_kernel_size: int = cfg.MODEL_CONFIG['cnn_kernel_size'],
        dropout: float = cfg.MODEL_CONFIG['dropout'],
        do_pool: bool = cfg.MODEL_CONFIG['do_pool'],
        stride_pos: str = cfg.MODEL_CONFIG['stride_pos'],
        stride_amt: int = cfg.MODEL_CONFIG['stride_amt'],
        **other_kwargs,
    ):
        down_stack = []
        in_shape = self.input_channels

        for _ in range(n_pre):
            down_stack.append(
                CGLLayer(in_shape, w_pre, cnn_kernel_size, type="cnn", dropout=dropout)
            )
            in_shape = down_stack[-1].output_size

        for _ in range(n_strided):
            stride = stride_amt
            pool = stride if (do_pool and stride > 1) else None 
            down_stack.append(
                CGLLayer(
                    in_shape,
                    w_strided,
                    cnn_kernel_size,
                    type="cnn",
                    stride=stride,
                    pool=pool,
                    stride_pos=stride_pos,
                    dropout=dropout,
                )
            )
            in_shape = down_stack[-1].output_size
            self.output_stride *= stride

        self.down_stack = nn.Sequential(*down_stack)

        lstm_stack = []
        for _ in range(n_l):
            lstm_stack.append(
                CGLLayer(
                    in_shape, 
                    w_l,      
                    cnn_kernel_size=cnn_kernel_size, 
                    type="lstm", 
                    dropout=dropout,
                )
            )
            in_shape = lstm_stack[-1].output_size

        self.lstm_stack = nn.Sequential(*lstm_stack)

        post_lstm_stack = []
        for _ in range(n_dense_post_l):
            post_lstm_stack.append(
                CGLLayer(in_shape, w_dense_post_l, kernel_size=1, type="cnn", dropout=dropout)
            )
            in_shape = post_lstm_stack[-1].output_size
        self.post_lstm_stack = nn.Sequential(*post_lstm_stack)
        
        self.feature_output_dim = in_shape

    def _forward(self, X: torch.Tensor, pad_mask=None) -> torch.Tensor:
        X = X.transpose(1, 2) 

        x_cnn = self.down_stack(X) 
        x_lstm = self.lstm_stack(x_cnn) 
        x_final_features_sequence = self.post_lstm_stack(x_lstm)
        
        extracted_features = x_final_features_sequence[:, :, -1] 
        
        return extracted_features

class RULRegressionHead(nn.Module):
    def __init__(self, feature_dim: int, dropout_rate: float = cfg.DROPOUT_RATE):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class RULFilterNet(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.feature_extractor = FilterNetFeatureExtractor(input_channels=input_channels, **cfg.MODEL_CONFIG)
        
        if cfg.MODEL_CONFIG['n_dense_post_l'] > 0:
            final_feature_dim_from_extractor = cfg.MODEL_CONFIG['w_dense_post_l']
        else:
            final_feature_dim_from_extractor = cfg.MODEL_CONFIG['w_l']

        self.regression_head = RULRegressionHead(
            feature_dim=final_feature_dim_from_extractor,
            dropout_rate=cfg.DROPOUT_RATE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)      
        rul_prediction = self.regression_head(features) 
        return rul_prediction