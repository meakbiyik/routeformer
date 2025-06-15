"""GPS backbone config definition."""
from dataclasses import dataclass, field

from routeformer.utils.config import BaseConfig


@dataclass
class GPSBackboneConfig(BaseConfig):
    """Config for GPS backbones."""

    seq_len: int
    label_len: int
    pred_len: int
    embed: str = "timeF"
    freq: str = "m"
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 512
    moving_avg: int = 25
    factor: int = 1
    distil: bool = True
    dropout: float = 0.1
    activation: str = "gelu"
    individual: bool = False
    # these are set after initialization by the parent config RouteformerConfig
    output_attention: bool = field(init=False)
    with_video: bool = field(init=False)
    with_gaze: bool = field(init=False)
    dense_prediction: bool = field(init=False)
    encoder_hidden_size: int = field(init=False)
    image_embedding_size: int = field(init=False)
    output_fps: int = field(init=False)
    dense_loss_ratio: float = field(init=False)
    discount_factor: dict = field(init=False)
    smart_decoder: bool = field(init=False)
    # This is a small hack for side experiments
    _enc_in: int = None
    _c_out: int = None

    @property
    def c_out(self) -> int:
        """Output dimension of the GPS backbone."""
        if self._c_out is not None:
            return self._c_out

        out = 2
        if not self.dense_prediction:
            return out

        return self.enc_in - 3 # angle, norm (speed), acceleration

    @property
    def enc_in(self) -> int:
        """Input dimension of the GPS backbone."""
        if self._enc_in is not None:
            return self._enc_in

        out = 2 + 3 # coords, plus angle, norm (speed), acceleration

        if not self.with_video:
            # We do not need to encode the GPS if we do not have video
            # since Informer already encodes the GPS, otherwise
            # we will need to sepeartely encode GPS and other features first
            return out

        out += self.encoder_hidden_size  # encoded visual

        return out

    @property
    def dec_in(self) -> int:
        """Input dimension of the GPS backbone."""
        return self.enc_in


@dataclass
class PatchTSTBackboneConfig(GPSBackboneConfig):
    """Config for PatchTST backbone."""

    fc_dropout: float = 0.1
    head_dropout: float = 0.0
    patch_len_ratio: int = 0.25
    stride_ratio: int = 0.125
    padding_patch: str = "end"
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    decomposition: bool = False
    kernel_size: int = 25

    @property
    def patch_len(self) -> int:
        """Patch length."""
        return int(self.patch_len_ratio * self.seq_len)

    @property
    def stride(self) -> int:
        """Stride."""
        return int(self.stride_ratio * self.seq_len)


@dataclass
class FEDFormerBackboneConfig(GPSBackboneConfig):
    """Config for FEDFormer backbone."""

    version: str = "Wavelets"
    mode_select: str = "random"
    modes: int = 32
    L: int = 0
    base: str = "legendre"
    cross_activation: str = "tanh"


@dataclass
class LinearBackboneConfig(GPSBackboneConfig):
    """Config for FEDFormer backbone."""

    kernel_size: int = 25
