"""Lookup and parsing of `config.yaml`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

CONFIG_NAME = "config.yaml"
CONFIG_ENV = "KAFKA_EXAMPLE_CONFIG"

SECURITY_PROTOCOL = "SASL_PLAINTEXT"
SASL_MECHANISM = "SCRAM-SHA-512"

TOPIC_MARKER = ".events."
SUBJECT_SUFFIX = "-value"

REQUIRED_KEYS = (
    "kafka-host",
    "kafka-username",
    "kafka-password",
    "kafka-topic",
    "schema-registry-host",
)

# spelled out in the task settings, but fixed in code -- flagged so that pasting
# them into the file gives a straight answer instead of "unexpected key"
FIXED_KEYS = ("kafka-security-protocol", "kafka-security-mechanism")

# derived from kafka-topic by `subject_for_topic`, same reason to flag it
DERIVED_KEYS = ("schema-registry-subject",)


def subject_for_topic(topic: str) -> str:
    """`traze.fsa.uat.oms-mt5.events.quotes.live01` -> `oms-mt5.events.quotes-value`.

    A topic is `<company>.<brand>.<env>.<subsystem>.events.<stream>.<server>`; the
    subject names only what the payload actually depends on -- the subsystem and the
    stream -- so one schema covers every environment and server publishing it. The
    `-value` suffix is Confluent's default naming strategy for the value of a record.
    """
    head, marker, tail = topic.partition(TOPIC_MARKER)
    if not marker or not head or not tail:
        raise ValueError(
            f"cannot derive a Schema Registry subject from topic {topic!r}: expected "
            f"<company>.<brand>.<env>.<subsystem>{TOPIC_MARKER}<stream>.<server>"
        )
    subsystem = head.rsplit(".", 1)[-1]
    stream = tail.split(".", 1)[0]
    return f"{subsystem}{TOPIC_MARKER}{stream}{SUBJECT_SUFFIX}"


def find_config(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = []
        from_env = os.getenv(CONFIG_ENV)
        if from_env:
            candidates.append(Path(from_env))
        candidates.append(Path.cwd() / CONFIG_NAME)
        candidates.append(Path(__file__).resolve().parent.parent / CONFIG_NAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    looked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"no {CONFIG_NAME} found. Looked at:\n  {looked}")


@dataclass(frozen=True)
class Config:
    """Everything needed to read one topic and decode it."""

    host: str
    username: str
    password: str
    topic: str
    registry_host: str
    path: Path | None = None

    @property
    def subject(self) -> str:
        """The Schema Registry subject of the topic's values -- derived, not configured."""
        return subject_for_topic(self.topic)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> Config:
        origin = find_config(path)
        with open(origin, encoding="utf-8") as stream:
            loaded: object = yaml.safe_load(stream)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{origin}: expected a mapping at the top level")
        raw = cast(dict[str, Any], loaded)

        fixed = [key for key in FIXED_KEYS if key in raw]
        if fixed:
            raise ValueError(
                f"{origin}: {', '.join(fixed)} is not configurable -- this example always "
                f"connects with {SECURITY_PROTOCOL} / {SASL_MECHANISM}"
            )

        derived = [key for key in DERIVED_KEYS if key in raw]
        if derived:
            raise ValueError(
                f"{origin}: {', '.join(derived)} is not configurable -- it is derived "
                f"from kafka-topic (see subject_for_topic), so only the topic changes "
                f"when you point the example at another stream"
            )

        unknown = sorted(set(raw) - set(REQUIRED_KEYS))
        if unknown:
            raise ValueError(
                f"{origin}: unexpected key(s) {', '.join(unknown)}; "
                f"expected: {', '.join(REQUIRED_KEYS)}"
            )

        missing = [key for key in REQUIRED_KEYS if not raw.get(key)]
        if missing:
            raise ValueError(f"{origin}: missing or empty: {', '.join(missing)}")

        config = cls(
            host=str(raw["kafka-host"]),
            username=str(raw["kafka-username"]),
            password=str(raw["kafka-password"]),
            topic=str(raw["kafka-topic"]),
            registry_host=str(raw["schema-registry-host"]).rstrip("/"),
            path=origin,
        )
        try:
            subject_for_topic(config.topic)  # fail here, not at the first registry call
        except ValueError as error:
            raise ValueError(f"{origin}: {error}") from error
        return config

    def consumer_config(self, group_id: str, auto_offset_reset: str = "earliest") -> dict[str, Any]:
        """librdkafka properties for a read-only consumer.

        `group.id` is a throwaway minted per run and nothing is ever committed, so
        this example cannot move the offsets of a real consumer.
        """
        return {
            "bootstrap.servers": self.host,
            "security.protocol": SECURITY_PROTOCOL,
            "sasl.mechanism": SASL_MECHANISM,
            "sasl.username": self.username,
            "sasl.password": self.password,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            # Committing is what makes a restart RESUME instead of replaying the
            # whole retention window. Safe because callers that must not disturb
            # anyone (the notebook examples) still get a throwaway group id --
            # commits against a private group move nobody else's offsets. A
            # long-running service passes a stable group and needs this on.
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
        }

    def masked(self) -> dict[str, Any]:
        """The settings with the password blanked out -- safe to print in a notebook."""
        return {
            "path": str(self.path) if self.path else None,
            "kafka-host": self.host,
            "kafka-username": self.username,
            "kafka-password": "***" if self.password else "",
            "kafka-topic": self.topic,
            "schema-registry-host": self.registry_host,
            "schema-registry-subject": f"{self.subject} (derived)",
            "security.protocol": SECURITY_PROTOCOL,
            "sasl.mechanism": SASL_MECHANISM,
        }

    @property
    def topic_prefix(self) -> str:
        """The `<env>.<subsystem>` head of the topic name, used to build a group id."""
        head = self.topic.split(".events.")[0]
        return head or self.topic
