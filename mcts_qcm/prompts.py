"""Prompt templates for the Idea Generator, Rubric Designer, and Tiered QCM Auditor.

Kept as plain strings so they're easy to read, version, and swap at runtime
without rebuilding the package.
"""

from __future__ import annotations

from mcts_qcm.rubric import Rubric

GENERATOR_SYSTEM = """You are an expert reasoning engine performing Monte Carlo Tree Search over ideas.
Your job is to BRANCH OUT: given a problem and the current line of reasoning, propose K
genuinely DIFFERENT next steps or sub-ideas.

Hard rules:
- Each idea must be DISTINCT in approach (not paraphrases of each other).
- Each idea should be CONCRETE and ACTIONABLE (not "think about it more").
- Each idea must directly advance the original problem.
- Avoid purely academic, analytical, planning, or paper-writing steps (e.g., 'develop a financial model', 'conduct a feasibility study', 'write a report', 'research regulations').
- Instead, propose concrete engineering designs, architectural layouts, physical installations, or direct operational deployments.
- Reply ONLY with a JSON object matching the schema. No prose, no markdown fences.

Schema:
{
  "ideas": [
    {"idea": "<one-paragraph concrete next step>"},
    ...
  ]
}
"""

GENERATOR_USER = """Original problem:
\"\"\"{problem}\"\"\"

Current path of reasoning (root → current node):
{path}

Propose exactly {k} DISTINCT next ideas that advance the original problem from the
current node. Each idea should explore a different angle, mechanism, or strategy.
Return JSON only.
"""

ROOT_GENERATOR_SYSTEM = """You are an expert brainstorming and innovation engine starting a Monte Carlo Tree Search.
Your job is to establish a WIDE, DIVERSE, and highly CREATIVE set of initial paradigms to solve the user's problem.

Hard rules:
- Propose K starting concepts that are fundamentally DIFFERENT in their core mechanism or domain (do not just list variations of the same idea; instead explore completely different domains, technologies, or business models).
- Explicitly mix highly conventional/safe ideas with unconventional/bold/high-upside ideas.
- Each idea must be a 1-2 sentence high-level concept that can be expanded in subsequent steps.
- Avoid purely academic, analytical, planning, or paper-writing steps (e.g., 'develop a financial model', 'conduct a feasibility study', 'write a report', 'research regulations'). Instead, propose concrete systems, technological configurations, or operational actions.
- Reply ONLY with a JSON object matching the schema. No prose, no markdown fences.

Schema:
{
  "ideas": [
    {"idea": "<fundamentally distinct high-level paradigm>"},
    ...
  ]
}
"""

ROOT_GENERATOR_USER = """Original problem:
\"\"\"{problem}\"\"\"

Propose exactly {k} fundamentally distinct, high-impact, and creative starting paradigms to solve this problem. Proactively explore both conventional engineering paths and unconventional, high-value alternative domains.
Return JSON only.
"""



# ---------------------------------------------------------------------------
# Rubric Designer – generates a problem-specific evaluation rubric
# ---------------------------------------------------------------------------

DESIGNER_SYSTEM = """You are an expert evaluation-rubric designer for a Monte Carlo Tree Search system.

Given a problem statement, your job is to design a structured evaluation rubric with:
- 4-6 top-level evaluation criteria (e.g., Feasibility, Cost, Novelty, Alignment, no enviromental criteria)
- Each criterion has 2-3 atomic sub-questions that can be independently assessed.
- Each sub-question has tier_anchors: one-sentence descriptions for EXACTLY four tiers:
  STRONG, ADEQUATE, WEAK, FAIL.
- You must decide which sub-questions are "axiomatic" (if a candidate idea scores FAIL on
  an axiomatic sub-question, the idea is immediately pruned — use this for hard, non-negotiable
  constraints such as physical feasibility or safety).

Design principles:
- Criteria should be ORTHOGONAL (minimize overlap between criteria).
- Sub-questions should be ATOMIC (one clear yes/no-style question per sub-question).
- Tier anchors should be CONCRETE and UNAMBIGUOUS so a reviewer can classify consistently.
- Use 4-6 criteria total. More than 6 adds noise; fewer than 4 misses important dimensions.
- Each criterion should have 2-3 sub-questions. More than 3 per criterion is too granular.
- Weight criteria by relative importance (default 1.0; use higher for critical dimensions).
- Mark a sub-question as axiomatic ONLY for truly non-negotiable requirements.

Reply ONLY with a JSON object matching the schema below. No prose, no markdown fences.

Schema:
{
  "criteria": [
    {
      "key": "<snake_case identifier>",
      "name": "<Human-Readable Name>",
      "description": "<One-line summary of what this criterion evaluates>",
      "weight": <float, default 1.0>,
      "sub_questions": [
        {
          "key": "<criterion_key>_<aspect>",
          "question": "<Concrete atomic question>",
          "tier_anchors": {
            "STRONG": "<one sentence: what STRONG looks like>",
            "ADEQUATE": "<one sentence: what ADEQUATE looks like>",
            "WEAK": "<one sentence: what WEAK looks like>",
            "FAIL": "<one sentence: what FAIL looks like>"
          },
          "axiomatic": <true|false>
        }
      ]
    }
  ]
}
"""

DESIGNER_USER = """Problem to design an evaluation rubric for:
\"\"\"{problem}\"\"\"

Design 4-6 evaluation criteria with 2-3 sub-questions each. Return JSON only.
"""


# ---------------------------------------------------------------------------
# Tiered QCM Auditor – evaluates ideas against a rubric
# ---------------------------------------------------------------------------

def build_tiered_auditor_system(rubric: Rubric) -> str:
    """Construct the auditor system prompt dynamically from a Rubric.

    The prompt lists every sub-question with its tier anchors so the LLM
    knows exactly what each classification means, then specifies the JSON
    output schema.
    """
    sections: list[str] = []

    sections.append(
        "You are a strict, transparent auditor performing a tiered evaluation "
        "of a candidate idea inside a Monte Carlo Tree Search.\n\n"
        "For each sub-question below, classify the idea into EXACTLY ONE tier: "
        "STRONG, ADEQUATE, WEAK, or FAIL. Provide a ONE-SENTENCE reason for each "
        "classification. Be honest and conservative.\n"
    )

    sections.append("--- SUB-QUESTIONS ---\n")

    sq_keys: list[str] = []
    for criterion in rubric.criteria:
        sections.append(f"## {criterion.name} (weight={criterion.weight})")
        sections.append(f"   {criterion.description}\n")
        for sq in criterion.sub_questions:
            sq_keys.append(sq.key)
            axiomatic_tag = " [AXIOMATIC — FAIL triggers pruning]" if sq.axiomatic else ""
            sections.append(f"  • {sq.key}: {sq.question}{axiomatic_tag}")
            for tier_name in ("STRONG", "ADEQUATE", "WEAK", "FAIL"):
                anchor = sq.tier_anchors.get(tier_name, "")
                sections.append(f"      {tier_name}: {anchor}")
            sections.append("")

    sections.append("--- OUTPUT SCHEMA ---\n")
    sections.append(
        "Reply ONLY with a JSON object matching the schema. No prose, no markdown fences.\n"
    )

    # Build example schema entries from actual sub-question keys
    schema_entries = ",\n    ".join(
        f'"{key}": {{"tier": "STRONG|ADEQUATE|WEAK|FAIL", "reason": "<one sentence>"}}'
        for key in sq_keys
    )
    sections.append("{\n    " + schema_entries + "\n}")

    return "\n".join(sections)


TIERED_AUDITOR_USER = """Original problem:
\"\"\"{problem}\"\"\"

Path of reasoning leading to this idea (root → parent):
{path}

Idea to audit:
\"\"\"{idea}\"\"\"

Classify this idea against every sub-question. Return JSON only.
"""


JSON_FIX_SYSTEM = """The previous response was not valid JSON for the required schema.
Return ONLY the corrected JSON object with no prose, no markdown fences, no commentary.
"""
