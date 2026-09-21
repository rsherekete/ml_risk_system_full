"""Schema Registry client and protobuf codegen.

Pulls a PROTOBUF schema out of the registry, writes it (plus every schema it
references) to disk, compiles it with the `protoc` bundled in `grpcio-tools`, and
imports the resulting `*_pb2` module -- no system `protoc`, no checked-in `.proto`.

    schema = generate(Config.load())
    Trade = schema.message_class()
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Sequence
from urllib.parse import quote

import requests
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.descriptor import FileDescriptor
from google.protobuf.message import Message

from .config import Config

DEFAULT_TIMEOUT = 15.0

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
PROTO_DIR = GENERATED_DIR / "proto"
PYTHON_DIR = GENERATED_DIR / "python"
MANIFEST = GENERATED_DIR / "manifest.json"


class SchemaRegistry:
    def __init__(self, host: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> Any:
        response = requests.get(f"{self.host}{path}", timeout=self.timeout)
        if response.status_code == 404:
            raise LookupError(f"{self.host}{path}: not found ({response.text.strip()})")
        response.raise_for_status()
        return response.json()

    def subjects(self) -> list[str]:
        return self.get("/subjects")

    def version(self, subject: str, version: int | str = "latest") -> dict[str, Any]:
        # a referenced schema is registered under its import path
        # (schemas/common/provider_enum.proto), whose slashes would otherwise be
        # read as path separators and 404
        return self.get(f"/subjects/{quote(subject, safe='')}/versions/{version}")

    def latest(self, subject: str) -> dict[str, Any]:
        return self.version(subject, "latest")

    def by_id(self, schema_id: int) -> dict[str, Any]:
        """A schema by the id embedded in a message -- carries no subject/version."""
        return self.get(f"/schemas/ids/{schema_id}")

    def subjects_for_id(self, schema_id: int) -> list[tuple[str, int]]:
        """The subject versions a schema id belongs to."""
        return [(entry["subject"], entry["version"]) for entry in self.get(f"/schemas/ids/{schema_id}/versions")]


def _require_protobuf(payload: dict[str, Any], what: str) -> str:
    schema_type = payload.get("schemaType", "AVRO")
    if schema_type != "PROTOBUF":
        raise ValueError(
            f"{what}: schemaType is {schema_type}, this example only decodes PROTOBUF"
        )
    schema = payload.get("schema")
    if not schema:
        raise ValueError(f"{what}: registry returned an empty schema")
    return str(schema)


def _module_name(subject: str) -> str:
    """`oms-mt4.events.trades-value` becomes `oms_mt4_events_trades`."""
    stem = re.sub(r"-(value|key)$", "", subject)
    stem = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower()
    if not stem or stem[0].isdigit():
        stem = f"schema_{stem}"
    return stem


def _write_proto(relative: str, text: str) -> Path:
    target = PROTO_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _fetch_references(
    registry: SchemaRegistry, references: Sequence[dict[str, Any]], seen: set[str]
) -> list[Path]:
    """Write every referenced schema at the import path the referring .proto uses."""
    written: list[Path] = []
    for reference in references or []:
        name, subject = reference.get("name"), reference.get("subject")
        version = reference.get("version", "latest")
        if not name or not subject or name in seen:
            continue
        seen.add(name)
        if name.startswith("google/protobuf/"):
            continue  # bundled with grpcio-tools
        payload = registry.version(subject, version)
        written.append(_write_proto(name, _require_protobuf(payload, f"reference {name}")))
        written.extend(_fetch_references(registry, payload.get("references", []), seen))
    return written


def _run_protoc(
    proto_files: Sequence[Path], descriptor_set: Path, python_out: bool = True
) -> None:
    from grpc_tools import protoc

    descriptor_set.parent.mkdir(parents=True, exist_ok=True)
    well_known = str(files("grpc_tools") / "_proto")
    sources = [str(path.relative_to(PROTO_DIR)).replace("\\", "/") for path in proto_files]

    argv = ["protoc", f"-I{PROTO_DIR}", f"-I{well_known}"]
    if python_out:
        PYTHON_DIR.mkdir(parents=True, exist_ok=True)
        argv.append(f"--python_out={PYTHON_DIR}")
    argv += [f"--descriptor_set_out={descriptor_set}", "--include_imports", *sources]

    code = protoc.main(argv)
    if code != 0:
        raise RuntimeError(f"protoc failed (exit {code}) on {sources}; see the output above")


def _walk_messages(
    messages: Sequence[descriptor_pb2.DescriptorProto], prefix: tuple[int, ...] = ()
) -> Iterator[tuple[tuple[int, ...], str]]:
    for position, message in enumerate(messages):
        path = prefix + (position,)
        yield path, message.name
        yield from _walk_messages(message.nested_type, path)


def _index_map(proto: descriptor_pb2.FileDescriptorProto) -> dict[tuple[int, ...], str]:
    return dict(_walk_messages(proto.message_type))


def _full_name(index_map: dict[tuple[int, ...], str], path: tuple[int, ...], package: str) -> str:
    name = index_map.get(path)
    if name is None:
        raise LookupError(
            f"no message at index path {list(path)}; "
            f"known: {[list(known) for known in index_map]}"
        )
    outer = [index_map[path[: depth + 1]] for depth in range(len(path) - 1)]
    return ".".join(filter(None, [package, *outer, name]))


@dataclass
class GeneratedSchema:
    """Generated classes for one subject, plus what is needed to decode its wire format."""

    subject: str
    schema_id: int
    version: int | None
    proto_path: Path
    descriptor_set: Path
    module: ModuleType
    reused_import: bool = False
    references: list[Path] = field(default_factory=list)

    @property
    def file_descriptor(self) -> FileDescriptor:
        return self.module.DESCRIPTOR

    @property
    def package(self) -> str:
        return self.file_descriptor.package

    @property
    def index_map(self) -> dict[tuple[int, ...], str]:
        """Index path -> message name, as Confluent message-indexes address them."""
        proto = descriptor_pb2.FileDescriptorProto()
        proto.ParseFromString(self.file_descriptor.serialized_pb)
        return _index_map(proto)

    @property
    def top_level_messages(self) -> list[str]:
        return [name for path, name in self.index_map.items() if len(path) == 1]

    def message_class(self, index_path: Sequence[int] = (0,)) -> type[Message]:
        """The class a message-index prefix points at; the default is the first message."""
        path = tuple(index_path) or (0,)
        full_name = _full_name(self.index_map, path, self.package)
        descriptor = self.file_descriptor.pool.FindMessageTypeByName(full_name)
        return message_factory.GetMessageClass(descriptor)


@dataclass
class PooledSchema:
    """A schema built into its own descriptor pool, addressed only by its id.

    Used for a schema id that is not the generated one: its package and message
    names collide with the already-imported module, and the global pool rejects
    a second definition of the same symbol.
    """

    schema_id: int
    file_name: str
    pool: descriptor_pool.DescriptorPool
    proto: descriptor_pb2.FileDescriptorProto

    @property
    def package(self) -> str:
        return self.proto.package

    @property
    def index_map(self) -> dict[tuple[int, ...], str]:
        return _index_map(self.proto)

    @property
    def top_level_messages(self) -> list[str]:
        return [name for path, name in self.index_map.items() if len(path) == 1]

    def message_class(self, index_path: Sequence[int] = (0,)) -> type[Message]:
        path = tuple(index_path) or (0,)
        full_name = _full_name(self.index_map, path, self.package)
        return message_factory.GetMessageClass(self.pool.FindMessageTypeByName(full_name))


def compile_by_id(registry: SchemaRegistry, schema_id: int) -> PooledSchema:
    """Fetch and compile a schema by the id read off a message."""
    payload = registry.by_id(schema_id)
    schema_text = _require_protobuf(payload, f"schema id {schema_id}")

    file_name = f"schema_{schema_id}.proto"
    proto_path = _write_proto(file_name, schema_text)
    references = _fetch_references(registry, payload.get("references", []), set())
    descriptor_set = GENERATED_DIR / f"schema_{schema_id}.desc"
    _run_protoc([proto_path, *references], descriptor_set, python_out=False)

    file_set = descriptor_pb2.FileDescriptorSet()
    file_set.ParseFromString(descriptor_set.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for file_proto in file_set.file:  # protoc emits dependencies first
        pool.Add(file_proto)

    main = next(file_proto for file_proto in file_set.file if file_proto.name == file_name)
    return PooledSchema(schema_id=schema_id, file_name=file_name, pool=pool, proto=main)


def generate(
    config: Config | None = None,
    subject: str | None = None,
    registry_host: str | None = None,
    force: bool = False,
) -> GeneratedSchema:
    """Fetch the newest schema for `subject` and return its generated classes.

    Protobuf registers descriptors globally on import, so a module already imported
    in this process is reused as-is; `reused_import` says so, and picking up a new
    schema version then needs a kernel restart.
    """
    if config is not None:
        registry_host = registry_host or config.registry_host
        subject = subject or config.subject
    if not registry_host or not subject:
        raise ValueError("pass a Config, or both registry_host and subject")

    registry = SchemaRegistry(registry_host)
    payload = registry.latest(subject)
    schema_text = _require_protobuf(payload, subject)
    schema_id, version = int(payload["id"]), payload.get("version")

    module_name = _module_name(subject)
    proto_path = _write_proto(f"{module_name}.proto", schema_text)
    references = _fetch_references(registry, payload.get("references", []), set())
    descriptor_set = GENERATED_DIR / f"{module_name}.desc"

    generated_py = PYTHON_DIR / f"{module_name}_pb2.py"
    if force or not generated_py.is_file() or not descriptor_set.is_file():
        _run_protoc([proto_path, *references], descriptor_set)

    if str(PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(PYTHON_DIR))
    reused = f"{module_name}_pb2" in sys.modules
    module = importlib.import_module(f"{module_name}_pb2")

    schema = GeneratedSchema(
        subject=subject,
        schema_id=schema_id,
        version=version,
        proto_path=proto_path,
        descriptor_set=descriptor_set,
        module=module,
        reused_import=reused,
        references=references,
    )
    _write_manifest(schema)
    return schema


def _write_manifest(schema: GeneratedSchema) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "subject": schema.subject,
                "schema_id": schema.schema_id,
                "version": schema.version,
                "package": schema.package,
                "proto": str(schema.proto_path.relative_to(GENERATED_DIR)).replace("\\", "/"),
                "references": [
                    str(path.relative_to(PROTO_DIR)).replace("\\", "/")
                    for path in schema.references
                ],
                "messages": {
                    ".".join(str(index) for index in path): name
                    for path, name in schema.index_map.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
