"""From https://github.com/MAZiqing/FEDformer."""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPSBackboneConfig
from .layers.AutoCorrelation import AutoCorrelationLayer
from .layers.AutoformerEncoderDecoder import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    SeasonalLayerNorm,
    series_decomp,
    series_decomp_multi,
)
from .layers.Embedding import DataEmbedding_wo_pos
from .layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform


class FEDformer(L.LightningModule):
    """FEDformer model as proposed in the paper.

    FEDformer performs the attention mechanism on frequency domain and
    achieved O(N) complexity
    """

    def __init__(self, configs: GPSBackboneConfig):
        """Initialize the model."""
        super().__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

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

        if configs.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes,
                ich=configs.d_model,
                base=configs.base,
                activation=configs.cross_activation,
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, configs.d_model, configs.n_heads
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
                        decoder_self_att, configs.d_model, configs.n_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att, configs.d_model, configs.n_heads
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
                dtype=torch.float32,
                device=x_enc.device,
            )
            .repeat(x_enc.shape[0])
            .view(x_enc.shape[0], -1, 1)
        )

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len)
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
