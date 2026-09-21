"""Kafka reader: consumes one topic, decodes protobuf, hands over JSON.

    with KafkaClient(Config.load()) as client:
        client.consume(lambda record: print(record.value), count=10)

Values carry the Confluent wire format -- a magic byte, the schema id, and the
message-indexes that say which message of the schema file was serialized:

    0x00 | schema id (4 bytes, big endian) | message-indexes | protobuf payload
"""

from __future__ import annotations

import os
import secrets
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from confluent_kafka import (
    TIMESTAMP_NOT_AVAILABLE,
    Consumer,
    KafkaError,
    KafkaException,
    Message as KafkaMessage,
)
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

from .config import Config
from .generate_proto import GeneratedSchema, PooledSchema, SchemaRegistry, compile_by_id, generate

MAGIC_BYTE = 0
HEADER_SIZE = 5


def _read_unsigned_varint(data: bytes, position: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if position >= len(data):
            raise ValueError("truncated varint in the message-index prefix")
        byte = data[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
        shift += 7
        if shift > 63:
            raise ValueError("varint too long in the message-index prefix")


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    """Message-indexes are zigzag-encoded, the way Kafka writes signed varints."""
    value, position = _read_unsigned_varint(data, position)
    return (value >> 1) ^ -(value & 1), position


@dataclass(frozen=True)
class WireHeader:
    schema_id: int
    index_path: tuple[int, ...]
    payload_start: int


def parse_wire_header(data: bytes) -> WireHeader:
    if len(data) <= HEADER_SIZE:
        raise ValueError(f"value is {len(data)} bytes, too short to be Confluent-framed")
    if data[0] != MAGIC_BYTE:
        raise ValueError(
            f"first byte is 0x{data[0]:02x}, expected 0x00 -- the value is not "
            f"Confluent-framed, so its schema cannot be identified"
        )

    schema_id = int.from_bytes(data[1:HEADER_SIZE], "big")
    count, position = _read_varint(data, HEADER_SIZE)
    if count == 0:
        # shorthand for [0], the first message of the file -- what every producer
        # of a single-message schema emits
        return WireHeader(schema_id, (0,), position)
    if count < 0:
        raise ValueError(f"negative message-index count ({count})")

    indexes: list[int] = []
    for _ in range(count):
        index, position = _read_varint(data, position)
        indexes.append(index)
    return WireHeader(schema_id, tuple(indexes), position)


class ProtobufDecoder:
    """Turns a Confluent-framed value into a dict.

    The schema id on the wire picks the classes, so a new version of the subject is
    picked up without regenerating. An id belonging to some *other* subject means the
    producer stamps the wrong one; that is reported once and the configured subject is
    used anyway, since decoding with a foreign schema yields plausible-looking
    nonsense rather than an error. `strict_schema_id` turns it into a failure.
    """

    def __init__(
        self,
        schema: GeneratedSchema,
        registry: SchemaRegistry | None = None,
        strict_schema_id: bool = False,
    ) -> None:
        self._schema = schema
        self._registry = registry
        self._strict = strict_schema_id
        self._resolved: dict[int, GeneratedSchema | PooledSchema] = {schema.schema_id: schema}
        self._classes: dict[tuple[int, tuple[int, ...]], type[Message]] = {}
        self._reported: set[int] = set()

    def decode(self, data: bytes) -> tuple[int, str, dict[str, Any]]:
        header = parse_wire_header(data)
        schema = self._schema_for(header.schema_id)
        message_class = self._class_for(schema, header.index_path)
        message = message_class()
        message.ParseFromString(data[header.payload_start :])
        return (
            schema.schema_id,
            message_class.DESCRIPTOR.name,
            MessageToDict(message, preserving_proto_field_name=True),
        )

    def _class_for(
        self, schema: GeneratedSchema | PooledSchema, index_path: tuple[int, ...]
    ) -> type[Message]:
        key = (schema.schema_id, index_path)
        cached = self._classes.get(key)
        if cached is None:
            cached = schema.message_class(index_path)
            self._classes[key] = cached
        return cached

    def _schema_for(self, schema_id: int) -> GeneratedSchema | PooledSchema:
        known = self._resolved.get(schema_id)
        if known is not None:
            return known

        if self._registry is None:
            resolved = self._mismatch(schema_id, "no registry was given to look it up")
        else:
            owners = self._registry.subjects_for_id(schema_id)
            if any(subject == self._schema.subject for subject, _ in owners):
                resolved = compile_by_id(self._registry, schema_id)
            else:
                owned = ", ".join(f"{s} v{v}" for s, v in owners) or "no subject"
                resolved = self._mismatch(schema_id, f"it belongs to {owned}")

        self._resolved[schema_id] = resolved
        return resolved

    def _mismatch(self, schema_id: int, reason: str) -> GeneratedSchema:
        report = (
            f"values carry schema id {schema_id}, which is not a version of "
            f"{self._schema.subject} ({reason}) -- the producer stamps the wrong id. "
            f"Decoding with {self._schema.subject} id {self._schema.schema_id} instead."
        )
        if self._strict:
            raise ValueError(report)
        if schema_id not in self._reported:
            self._reported.add(schema_id)
            warnings.warn(report, stacklevel=3)
        return self._schema


@dataclass(frozen=True)
class Record:
    """One Kafka message: its coordinates plus the decoded value."""

    topic: str
    partition: int
    offset: int
    timestamp: datetime | None
    key: str | None
    schema_id: int | None
    message: str | None
    value: dict[str, Any] | None


def _decode_key(key: bytes | None) -> str | None:
    if key is None:
        return None
    try:
        return key.decode("utf-8")
    except UnicodeDecodeError:
        return key.hex()


def _decode_timestamp(message: KafkaMessage) -> datetime | None:
    kind, milliseconds = message.timestamp()
    if kind == TIMESTAMP_NOT_AVAILABLE:
        return None
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)


class KafkaClient:
    """Reads one topic and delivers decoded records to a callback."""

    def __init__(
        self,
        config: Config,
        schema: GeneratedSchema | None = None,
        auto_offset_reset: str = "latest",
        group_id: str | None = None,
        strict_schema_id: bool = False,
    ) -> None:
        self.config = config
        self.schema = schema if schema is not None else generate(config)
        self.registry = SchemaRegistry(config.registry_host)
        self.decoder = ProtobufDecoder(self.schema, self.registry, strict_schema_id)
        self.group_id = group_id or f"{config.topic_prefix}.{os.getpid()}.{secrets.token_hex(4)}"
        self._consumer = Consumer(config.consumer_config(self.group_id, auto_offset_reset))
        self._subscribed = False

    def __enter__(self) -> KafkaClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._consumer.close()

    def partitions(self, timeout: float = 15.0) -> list[int]:
        """Broker round-trip that also proves the credentials and the topic name."""
        metadata = self._consumer.list_topics(self.config.topic, timeout=timeout)
        topic = metadata.topics[self.config.topic]
        if topic.error is not None:
            raise self._error(topic.error)
        return sorted(topic.partitions)

    def visible_topics(self, timeout: float = 15.0) -> list[str]:
        """Every topic this principal may see -- the way to check a topic name."""
        return sorted(self._consumer.list_topics(timeout=timeout).topics)

    def consume(
        self,
        on_message: Callable[[Record], None],
        count: int = 0,
        timeout: float | None = None,
        poll_timeout: float = 1.0,
    ) -> int:
        """Read `count` messages, or stream until interrupted when `count` is 0.

        Blocks while the topic is quiet: with the default `latest` offset reset only
        messages produced after subscribing are seen. `timeout` bounds that wait,
        `Ctrl-C` ends it.
        """
        if not self._subscribed:
            self._consumer.subscribe([self.config.topic])
            self._subscribed = True

        delivered = 0
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while count == 0 or delivered < count:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                message = self._consumer.poll(poll_timeout)
                if message is None:
                    continue
                error = message.error()
                if error is not None:
                    self._raise(error)
                    continue
                record = self._record(message)
                delivered += 1
                on_message(record)
        except KeyboardInterrupt:
            pass
        return delivered

    def _raise(self, error: KafkaError) -> None:
        if error.code() == KafkaError._PARTITION_EOF or error.retriable():
            return
        raise self._error(error)

    def _error(self, error: KafkaError) -> Exception:
        if error.code() == KafkaError.TOPIC_AUTHORIZATION_FAILED:
            # a misspelt topic looks exactly like a missing ACL: the broker will not
            # admit that a topic the principal cannot read exists at all
            return PermissionError(
                f"{self.config.username} cannot read topic {self.config.topic!r} -- "
                f"either the name is wrong or the ACL is missing. "
                f"KafkaClient.visible_topics() lists what this user may see. ({error.str()})"
            )
        if error.code() == KafkaError.GROUP_AUTHORIZATION_FAILED:
            return PermissionError(
                f"{self.config.username} may not join group {self.group_id!r}: {error.str()}"
            )
        if error.code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
            return LookupError(f"no topic {self.config.topic!r} on {self.config.host}")
        return KafkaException(error)

    def _record(self, message: KafkaMessage) -> Record:
        payload = message.value()
        schema_id, name, value = (None, None, None)
        if payload is not None:
            schema_id, name, value = self.decoder.decode(payload)
        return Record(
            topic=message.topic() or self.config.topic,
            partition=message.partition() or 0,
            offset=message.offset() or 0,
            timestamp=_decode_timestamp(message),
            key=_decode_key(message.key()),
            schema_id=schema_id,
            message=name,
            value=value,
        )
