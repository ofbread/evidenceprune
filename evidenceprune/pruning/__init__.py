"""Pruning: split a page into sentences, score them with a student model, keep the relevant ones."""
from .pruner import DEFAULT_THRESHOLDS, Document, Pruned, Pruner
from .sentences import sentences, windows

__all__ = ["Pruner", "Pruned", "Document", "DEFAULT_THRESHOLDS", "sentences", "windows"]
