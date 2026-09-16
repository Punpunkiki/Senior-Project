"""app/knowledge.py -- loads the disease knowledge base and validates that it
covers every class the model can predict.

The KB is keyed by `class_name`, which must match config.yaml -> data.classes
exactly. A mismatch means the bot could predict a class it cannot describe,
so it is caught at startup rather than in front of a farmer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.diagnosis import NOT_DURIAN_CLASS

DEFAULT_KB_PATH = Path(__file__).resolve().parent / "data" / "diseases.json"


@dataclass
class Disease:
    class_name: str
    slug: str
    name_th: str
    name_en: str
    pathogen: Optional[str]
    type: Optional[str]
    severity: str
    affected_parts: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    causes_conditions: List[str] = field(default_factory=list)
    immediate_actions: List[str] = field(default_factory=list)
    chemical_options: List[Dict[str, str]] = field(default_factory=list)
    prevention: List[str] = field(default_factory=list)
    when_to_call_expert: str = ""
    images: List[str] = field(default_factory=list)
    references: List[Dict[str, str]] = field(default_factory=list)
    reviewed_by_expert: bool = False
    pending_expert_input: bool = False

    @property
    def is_healthy(self) -> bool:
        return self.type == "healthy"


class KnowledgeBase:
    def __init__(self, diseases: List[Disease]):
        self._by_class = {d.class_name: d for d in diseases}
        self._by_slug = {d.slug: d for d in diseases}

    def by_class(self, class_name: str) -> Optional[Disease]:
        return self._by_class.get(class_name)

    def by_slug(self, slug: str) -> Optional[Disease]:
        return self._by_slug.get(slug)

    def all(self) -> List[Disease]:
        return list(self._by_class.values())

    def ready(self) -> List[Disease]:
        """Entries that actually have content to show."""
        return [d for d in self._by_class.values() if not d.pending_expert_input]

    def pending(self) -> List[Disease]:
        return [d for d in self._by_class.values() if d.pending_expert_input]

    def missing_for(self, model_classes: List[str]) -> List[str]:
        """Model classes with no KB entry. `not_durian` is excluded: it is a
        reject class handled by the tier logic, not a disease."""
        return [
            c
            for c in model_classes
            if c != NOT_DURIAN_CLASS and c not in self._by_class
        ]


def _to_disease(raw: Dict[str, Any]) -> Disease:
    known = {f for f in Disease.__dataclass_fields__}
    return Disease(**{k: v for k, v in raw.items() if k in known})


@lru_cache(maxsize=4)
def load_knowledge_base(path: str | Path = DEFAULT_KB_PATH) -> KnowledgeBase:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return KnowledgeBase([_to_disease(d) for d in raw["diseases"]])
