"""From https://github.com/MAZiqing/FEDformer."""
import lightning as L
import torch
import torch.nn as nn

from .config import GPSBackboneConfig
from .layers.Embedding import DataEmbedding
from .layers.SelfAttentionFamily import AttentionLayer, ProbAttention
from .layers.TransformerEncoderDecoder import (
    ConvLayer,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)


class Informer(L.LightningModule):
    """Informer with Propspare attention in O(LlogL) complexity."""

    def __init__(self, configs: GPSBackboneConfig):
        """Initialize the model."""
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.smart_decoder = configs.smart_decoder

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
                        ProbAttention(
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
            [ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)]
            if configs.distil
            else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
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

        if self.smart_decoder:
            x_dec = torch.cat(
                (
                    x_enc,
                    x_enc[:, -1:, :].repeat(
                        1,
                        self.pred_len,
                        1,
                    ),
                ),
                dim=1,
            )
        else:
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


class TriangularCausalMask:
    """Triangular causal mask."""

    def __init__(self, B, L, device="cpu"):
        """Initialize the mask."""
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """Get the mask."""
        return self._mask


class ProbMask:
    """Probabilistic mask."""

    def __init__(self, B, H, L, index, scores, device="cpu"):
        """Initialize the mask."""
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """Get the mask."""
        return self._mask
