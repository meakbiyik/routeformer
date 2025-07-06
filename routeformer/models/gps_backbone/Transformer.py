"""From https://github.com/MAZiqing/FEDformer."""
import lightning as L
import torch
import torch.nn as nn

from .config import GPSBackboneConfig
from .layers.Embedding import DataEmbedding
from .layers.SelfAttentionFamily import AttentionLayer, FullAttention
from .layers.TransformerEncoderDecoder import Decoder, DecoderLayer, Encoder, EncoderLayer


class Transformer(L.LightningModule):
    """Vanilla Transformer with O(L^2) complexity."""

    def __init__(self, configs: GPSBackboneConfig):
        """Initialize the model."""
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.dec_embedding = DataEmbedding(
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
                    AttentionLayer(
                        FullAttention(
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
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
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
        x_dec = torch.cat(
            (
                x_enc,
                torch.zeros(
                    x_enc.shape[0],
                    self.pred_len,
                    x_enc.shape[-1],
                    device=x_enc.device,
                    dtype=torch.float32,
                ),
            ),
            dim=1,
        )
        x_mark_dec = (
            torch.arange(x_enc.shape[1] + self.pred_len, device=x_enc.device, dtype=torch.float32)
            .repeat(x_dec.shape[0])
            .view(x_dec.shape[0], -1, 1)
        )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]
