"""Project constants and enumerations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionDefinition:
    """Definition of one antibiotic action."""

    code: str
    label: str
    aliases: tuple[str, ...]


ACTIONS: tuple[ActionDefinition, ...] = (
    ActionDefinition(
        code="3gc",
        label="Third-generation cephalosporin",
        aliases=(
            "3gc",
            "third-generation cephalosporin",
            "third generation cephalosporin",
            "3rd generation cephalosporin",
            "cefotaxime",
            "ceftriaxone",
            "ceftazidime",
        ),
    ),
    ActionDefinition(
        code="fq",
        label="Fluoroquinolone",
        aliases=(
            "fq",
            "fluoroquinolone",
            "ciprofloxacin",
            "levofloxacin",
        ),
    ),
    ActionDefinition(
        code="carb",
        label="Carbapenem",
        aliases=(
            "carb",
            "carbapenem",
            "imipenem",
            "meropenem",
            "ertapenem",
        ),
    ),
)

ACTION_INDEX = {action.code: index for index, action in enumerate(ACTIONS)}
STATE_COUNT = 8
