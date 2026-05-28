"""MCTS QCM Reasoning Engine.

An MCTS framework where LLM-driven tiered QCM (Multiple Choice Question) audits
replace the neural value network used in AlphaGo-style tree search.
"""

from mcts_qcm.auditor import AuditResult, QCMAuditor
from mcts_qcm.config import DEFAULT_GEMINI_FLASH, MCTSConfig
from mcts_qcm.designer import QCMDesigner
from mcts_qcm.generator import IdeaGenerator
from mcts_qcm.node import Node
from mcts_qcm.rubric import Rubric
from mcts_qcm.search import MCTS

__all__ = [
    "MCTS",
    "MCTSConfig",
    "DEFAULT_GEMINI_FLASH",
    "Node",
    "IdeaGenerator",
    "QCMAuditor",
    "AuditResult",
    "QCMDesigner",
    "Rubric",
]

__version__ = "0.1.0"
