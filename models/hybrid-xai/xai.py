"""Module 3 — Explainability (XAI) for COSMIC measurements.

Generates human-readable traceability reports from NormalizedRequirement
instances, mapping each data movement back to the original text and
producing Mermaid diagrams.
"""
from __future__ import annotations

from .schema import (
    DataMovement,
    FunctionalProcess,
    MovementType,
    NormalizedRequirement,
)
from .rule_engine import RuleEngineResult


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def generate_trace_report(nr: NormalizedRequirement, result: RuleEngineResult) -> str:
    """Generate a human-readable measurement trace."""
    lines: list[str] = []
    lines.append(f"# COSMIC Measurement Trace — {nr.id}")
    lines.append(f"")
    lines.append(f"**Source format:** {nr.source_format.value}")
    lines.append(f"**Total CFP:** {result.total_cfp}")
    lines.append(f"**Boundary:** {nr.boundary.software_name}")
    lines.append(f"**Functional users:** {', '.join(nr.boundary.functional_users)}")
    lines.append(f"")

    for pr in result.process_results:
        lines.append(f"## Process: {pr.process_name} ({pr.cfp} CFP)")
        lines.append(f"")

        e_count = sum(1 for m in pr.movements if m.type == MovementType.E)
        x_count = sum(1 for m in pr.movements if m.type == MovementType.X)
        r_count = sum(1 for m in pr.movements if m.type == MovementType.R)
        w_count = sum(1 for m in pr.movements if m.type == MovementType.W)
        lines.append(f"**Breakdown:** {e_count}E + {r_count}R + {w_count}W + {x_count}X = {pr.cfp} CFP")
        lines.append(f"")

        lines.append(f"| # | Type | Data Group | OOI | Functional User | Flags |")
        lines.append(f"|---|------|-----------|-----|-----------------|-------|")
        for i, m in enumerate(pr.movements, 1):
            flags = []
            if m.is_triggering_entry:
                flags.append("triggering")
            if m.is_error_confirmation:
                flags.append("error/confirm")
            lines.append(
                f"| {i} | **{m.type.value}** | {m.data_group} | {m.object_of_interest} "
                f"| {m.functional_user or '—'} | {', '.join(flags) or '—'} |"
            )
        lines.append(f"")

        if pr.corrections:
            lines.append(f"**M2 corrections:**")
            for c in pr.corrections:
                lines.append(f"- {c}")
            lines.append(f"")

        if pr.removed:
            lines.append(f"**Removed movements:**")
            for reason, dm in pr.removed:
                lines.append(f"- [{reason}] {dm.type.value}: {dm.data_group}")
            lines.append(f"")

    # Sub-process traceability
    for fp in nr.functional_processes:
        if fp.sub_processes:
            lines.append(f"## Traceability: {fp.name}")
            lines.append(f"")
            lines.append(f"| Step | Actor | Original Text | Direction | Movement |")
            lines.append(f"|------|-------|---------------|-----------|----------|")
            for sp in fp.sub_processes:
                d = sp.direction.value if sp.direction else "—"
                mt = sp.data_movement.type.value if sp.data_movement else "—"
                lines.append(
                    f"| {sp.step_id} | {sp.actor} | {sp.action_text[:60]} | {d} | {mt} |"
                )
            lines.append(f"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mermaid diagram
# ---------------------------------------------------------------------------

def generate_mermaid(nr: NormalizedRequirement, result: RuleEngineResult) -> str:
    """Generate a Mermaid flowchart of data movements."""
    lines: list[str] = []
    lines.append("```mermaid")
    lines.append("flowchart LR")

    for pr in result.process_results:
        safe_name = pr.process_name.replace(" ", "_").replace('"', "")
        lines.append(f'    subgraph {safe_name}["{pr.process_name} ({pr.cfp} CFP)"]')

        for i, m in enumerate(pr.movements):
            node_id = f"{safe_name}_{i}"
            label = f"{m.type.value}: {m.data_group[:30]}"
            shape = _movement_shape(m.type)
            lines.append(f'        {node_id}{shape[0]}"{label}"{shape[1]}')

        # Link movements sequentially
        ids = [f"{safe_name}_{i}" for i in range(len(pr.movements))]
        for j in range(len(ids) - 1):
            lines.append(f"        {ids[j]} --> {ids[j+1]}")

        lines.append("    end")

    # Boundary context
    if nr.boundary.functional_users:
        fu = nr.boundary.functional_users[0]
        lines.append(f'    FU(("{fu}"))' )
        for pr in result.process_results:
            safe_name = pr.process_name.replace(" ", "_").replace('"', "")
            entries = [i for i, m in enumerate(pr.movements) if m.type == MovementType.E]
            exits = [i for i, m in enumerate(pr.movements) if m.type == MovementType.X]
            for idx in entries:
                lines.append(f"    FU -->|Entry| {safe_name}_{idx}")
            for idx in exits:
                lines.append(f"    {safe_name}_{idx} -->|Exit| FU")

    if nr.boundary.persistent_stores:
        lines.append(f'    DB[("{nr.boundary.persistent_stores[0]}")]')
        for pr in result.process_results:
            safe_name = pr.process_name.replace(" ", "_").replace('"', "")
            reads = [i for i, m in enumerate(pr.movements) if m.type == MovementType.R]
            writes = [i for i, m in enumerate(pr.movements) if m.type == MovementType.W]
            for idx in reads:
                lines.append(f"    DB -->|Read| {safe_name}_{idx}")
            for idx in writes:
                lines.append(f"    {safe_name}_{idx} -->|Write| DB")

    lines.append("```")
    return "\n".join(lines)


def _movement_shape(mt: MovementType) -> tuple[str, str]:
    """Return Mermaid node shape delimiters based on movement type."""
    shapes = {
        MovementType.E: ("[/", "/]"),
        MovementType.X: ("[\\", "\\]"),
        MovementType.R: ("[(", ")]"),
        MovementType.W: ("[[", "]]"),
    }
    return shapes.get(mt, ("[", "]"))
