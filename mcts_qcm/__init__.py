"""MCTS QCM Reasoning Engine.

An MCTS framework where LLM-driven QCM (Multiple Choice Question) audits replace the
neural value network used in AlphaGo-style tree search.
"""

from mcts_qcm.auditor import QCMAuditor, QCMResult
from mcts_qcm.config import MCTSConfig
from mcts_qcm.generator import IdeaGenerator
from mcts_qcm.node import Node
from mcts_qcm.search import MCTS

__all__ = [
    "MCTS",
    "MCTSConfig",
    "Node",
    "IdeaGenerator",
    "QCMAuditor",
    "QCMResult",
]

__version__ = "0.1.0"
