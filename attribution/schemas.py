from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TokenAttribution:
    rank: int
    sequence_position: int
    token: str
    token_id: int
    change_type: str
    raw_score: float
    normalized_score: float
    layer_strategy: str
    observed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenAttributionResult:
    commit_id: str
    model_name: str
    checkpoint_id: str
    prediction_score: float
    predicted_label: int
    threshold: float
    attribution_method: str
    attention_strategy: str
    explanation_scope: str
    attribution_is_class_specific: bool
    ranked_tokens: List[TokenAttribution]
    observed_sequence_tokens: int
    observed_code_tokens: int
    observed_added_tokens: int
    observed_removed_tokens: int
    pre_truncation_content_tokens: int
    truncated_content_tokens: int
    truncated: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "succeeded"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["ranked_tokens"] = [token.to_dict() for token in self.ranked_tokens]
        return result
