"""Pydantic data models for the authority-projection synthetic data pipeline."""

import hashlib
import json
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


AgeBucket = Literal["20s", "30s", "40s", "50s+"]
CommunicationStyle = Literal["terse", "verbose", "casual", "formal"]
PriorAIExperience = Literal["none", "casual", "heavy"]
LifeContext = Literal["stable", "mild_stress", "major_transition"]
ConversationalGoal = Literal[
    "seeking_information",
    "brainstorming",
    "venting",
    "validation_seeking",
    "pressure_testing",
]
Tier = Literal["none", "somewhat", "strongly"]
ArcShape = Literal["stable", "escalating", "de_escalating", "mixed"]
LexicalTwinKind = Literal["match", "peer_voice", "submission_voice"]


class Persona(BaseModel):
    age_bucket: AgeBucket
    communication_style: CommunicationStyle
    prior_ai_experience: PriorAIExperience
    life_context: LifeContext
    conversational_goal: ConversationalGoal

    def to_prompt_description(self) -> str:
        return (
            f"User is in their {self.age_bucket}, "
            f"with a {self.communication_style} communication style. "
            f"They have {self.prior_ai_experience} prior experience with AI assistants. "
            f"Current life context: {self.life_context.replace('_', ' ')}. "
            f"Conversational goal in this chat: {self.conversational_goal.replace('_', ' ')}."
        )


class GenerationSeed(BaseModel):
    """Specification for one conversation to generate. seed_id used for dedup/resume."""

    seed_id: str
    topic: str
    persona: Persona
    target_tier: Tier
    arc_shape: ArcShape
    lexical_twin_kind: LexicalTwinKind
    n_turns: int = Field(ge=3, le=5)

    @classmethod
    def make(
        cls,
        topic: str,
        persona: Persona,
        target_tier: Tier,
        arc_shape: ArcShape,
        lexical_twin_kind: LexicalTwinKind,
        n_turns: int,
        salt: str = "",
    ) -> "GenerationSeed":
        payload = json.dumps(
            {
                "topic": topic,
                "persona": persona.model_dump(),
                "target_tier": target_tier,
                "arc_shape": arc_shape,
                "lexical_twin_kind": lexical_twin_kind,
                "n_turns": n_turns,
                "salt": salt,
            },
            sort_keys=True,
        )
        seed_id = hashlib.md5(payload.encode()).hexdigest()
        return cls(
            seed_id=seed_id,
            topic=topic,
            persona=persona,
            target_tier=target_tier,
            arc_shape=arc_shape,
            lexical_twin_kind=lexical_twin_kind,
            n_turns=n_turns,
        )


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TurnLabel(BaseModel):
    turn_index: int = Field(description="Index among USER turns; 0 = first user turn.")
    label: Tier
    rationale: str


class Conversation(BaseModel):
    """A generated conversation, optionally with per-user-turn labels."""

    model_config = {"extra": "allow"}

    seed_id: str
    topic: str
    persona: Persona
    target_tier: Tier
    arc_shape: Optional[ArcShape] = None
    lexical_twin_kind: Optional[LexicalTwinKind] = None
    uses_lexical_twin: Optional[bool] = None  # legacy v0 field; superseded by lexical_twin_kind
    conversation: List[Turn]
    turn_labels: Optional[List[TurnLabel]] = None


class GenerationOutput(BaseModel):
    """Schema the generator LLM is asked to return."""

    conversation: List[Turn]


class LabelingOutput(BaseModel):
    """Schema the labeler LLM is asked to return."""

    turns: List[TurnLabel]
