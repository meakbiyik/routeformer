"""From https://github.com/MAZiqing/FEDformer."""
# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import lightning as L
import torch
import torch.nn as nn

from .config import GPSBackboneConfig
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.AutoformerEncoderDecoder import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    SeasonalLayerNorm,
    series_decomp,
)
from .layers.Embedding import DataEmbedding_wo_pos


class Autoformer(L.LightningModule):
    """Autoformer is a series-wise transformer.

    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs: GPSBackboneConfig):
        """Initialize the model."""
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp(kernel_size[0])
        else:
            self.decomp = series_decomp(kernel_size)
        self.trend_projection = nn.Linear(configs.enc_in, configs.c_out, bias=True)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=SeasonalLayerNorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=SeasonalLayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data. Shape: [Batch, Input length, Channel]

        Returns
        -------
        torch.Tensor
            Prediction. Shape: [Batch, Prediction length, Channel]
        """
        x_enc = x
        x_mark_enc = (
            torch.arange(x_enc.shape[1], device=x_enc.device, dtype=torch.float32)
            .repeat(x_enc.shape[0])
            .view(x_enc.shape[0], -1, 1)
        )
        x_mark_dec = (
            torch.arange(
                x_enc.shape[1] - self.label_len,
                x_enc.shape[1] + self.pred_len,
                device=x_enc.device,
                dtype=torch.float32,
            )
            .repeat(x_enc.shape[0])
            .view(x_enc.shape[0], -1, 1)
        )

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        trend_init = self.trend_projection(trend_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]
