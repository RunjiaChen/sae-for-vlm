from .dictionary import AutoEncoder, GatedAutoEncoder, JumpReluAutoEncoder
try:
    from .buffer import ActivationBuffer
except ImportError:  # nnsight is only needed for training-time activation buffering
    ActivationBuffer = None
