"""claimprune — claim-conditioned context pruning with distilled small models."""
from .pruner import DEFAULT_THRESHOLDS, Document, Pruned, Pruner
from .sentences import sentences, windows

__all__ = ["Pruner", "Pruned", "Document", "DEFAULT_THRESHOLDS", "sentences", "windows"]
__version__ = "0.1.0"
