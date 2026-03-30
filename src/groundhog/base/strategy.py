"""Strategy — an action that moves the optimizer's state forward.

A strategy is any deliberate step the optimizer can take. It does not have to
produce a new candidate — planning, analysing, or updating learnings all count.

Receives the Toolkit (available capabilities) and an optional config (what to
do this time). The strategy owns the full loop: select prior, generate code,
evaluate, record the attempt. The optimizer just calls it and logs the result.

Returns a Dict[str, Any] for debug/logging. Nothing in the system depends on
the return value — strategies are free to put whatever they want in there.

The business logic layer — deliberately unconstrained. Other abstractions
(Toolkit, Scorer, AttemptHistory) exist to be clean and stable. Strategies
are where implementation details, flexibility, and messiness belong.

Composed method pattern: __call__ should read like a story — each step is a
clearly named method call. Implementation details live in the step methods.
This makes strategies readable and auditable at a glance.

Each strategy has a Config dataclass that declares all configurable parameters
with types, defaults, and descriptions. Any tool (UI, optimizer, LLM) can
inspect Config fields to discover what's configurable.

Config flows: class defaults → constructor → call-time override.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict


def param(default, description: str, **kwargs):
    """Declare a config parameter with a description.

    Usage:
        max_retries: int = param(3, "Max retry attempts on evaluation failure")
    """
    return field(default=default, metadata={"description": description, **kwargs})


@dataclass
class StrategyConfig:
    """Base config. Strategies subclass this with their own fields.

    Each field should use param() to include a description:
        my_field: int = param(10, "What this field controls")
    """

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        """Construct from dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in known})

    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Return {field_name: {type, default, value, description}} for all params."""
        result = {}
        for f in fields(self):
            result[f.name] = {
                "type": f.type,
                "default": f.default,
                "value": getattr(self, f.name),
                "description": f.metadata.get("description", ""),
            }
        return result


class Strategy(ABC):
    """Interface for a strategy.

    Every strategy must define a Config dataclass (subclass of StrategyConfig)
    that declares its configurable parameters.

    Config resolution: class defaults → constructor config → call-time config.

    Use composed method pattern: __call__ reads like a story, details in steps.
    """

    Config = StrategyConfig

    def __init__(self, config=None, **kwargs):
        """Create strategy with optional config override.

        Args:
            config: Config instance, dict, or None for defaults
            **kwargs: individual config fields (convenience)
        """
        if isinstance(config, self.Config):
            self.config = config
        elif isinstance(config, dict):
            self.config = self.Config.from_dict(config)
        elif kwargs:
            self.config = self.Config.from_dict(kwargs)
        else:
            self.config = self.Config()

    def _resolve_config(self, call_config=None):
        """Merge call-time config over instance config.

        Returns a Config instance with call-time values overriding instance values.
        """
        if call_config is None:
            return self.config
        if isinstance(call_config, dict):
            return replace(self.config, **{
                k: v for k, v in call_config.items()
                if k in {f.name for f in fields(self.config)}
            })
        if isinstance(call_config, self.Config):
            return call_config
        return self.config

    @abstractmethod
    def __call__(self, toolkit, config=None) -> Dict[str, Any]: ...
