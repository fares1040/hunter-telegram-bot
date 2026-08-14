"""Hunter Bot — Data Confidence Tracker"""
from dataclasses import dataclass, field
from typing import List
from enum import Enum


class DataQuality(Enum):
    REAL = "REAL"
    PROXY = "PROXY"
    MISSING = "MISSING"


@dataclass
class ConfidenceField:
    name: str
    quality: DataQuality
    weight: float = 1.0
    note: str = ""


@dataclass
class DataConfidenceReport:
    ticker: str
    fields: List[ConfidenceField] = field(default_factory=list)

    def add(self, name: str, quality: DataQuality, weight: float = 1.0, note: str = ""):
        self.fields.append(ConfidenceField(name, quality, weight, note))

    @property
    def score(self) -> int:
        if not self.fields:
            return 0
        total_weight = sum(f.weight for f in self.fields)
        if total_weight == 0:
            return 0
        weighted_sum = sum(
            (1.0 if f.quality == DataQuality.REAL else 0.5 if f.quality == DataQuality.PROXY else 0.0)
            * f.weight
            for f in self.fields
        )
        return int((weighted_sum / total_weight) * 100)

    @property
    def missing_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.quality == DataQuality.MISSING]

    @property
    def proxy_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.quality == DataQuality.PROXY]

    def summary(self) -> str:
        return (
            f"DataConfidence({self.ticker}): {self.score}% | "
            f"REAL={len([f for f in self.fields if f.quality==DataQuality.REAL])}, "
            f"PROXY={len(self.proxy_fields)}, "
            f"MISSING={len(self.missing_fields)}"
        )
