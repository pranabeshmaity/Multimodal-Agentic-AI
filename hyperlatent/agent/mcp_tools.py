"""MCP-style tool definitions and registry.

Defines a minimal `Tool` dataclass plus a `ToolRegistry` that serializes its
contents to the Anthropic Model Context Protocol (MCP) schema shape:
`{name, description, inputSchema}`. Ships with two reference tools:

- `python_sandbox`: runs Python source in an isolated tempdir via
  `subprocess` with a wall-clock timeout and best-effort POSIX resource
  limits. This is NOT a security boundary; see `_set_rlimits` below.
- `file_system`: read/write/list within a caller-supplied root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:  # POSIX-only
    import resource as _resource  # type: ignore

    _HAS_RESOURCE = True
except Exception:  # pragma: no cover - Windows
    _resource = None  # type: ignore
    _HAS_RESOURCE = False


# ---------------------------------------------------------------------- types
@dataclass
class Tool:
    """A single callable tool in the agent's action space.

    Attributes:
        name: Machine-readable identifier, matches the MCP `name` field.
        description: Human-readable description, matches the MCP
            `description` field.
        json_schema: JSON Schema describing the tool's input, matches the
            MCP `inputSchema` field.
        handler: Python callable invoked with the validated input dict.
    """

    name: str
    description: str
    json_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]] = field(repr=False)

    def to_mcp(self) -> Dict[str, Any]:
        """Return the MCP-compatible dict representation of this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.json_schema,
        }

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the tool's handler with `payload`."""
        return self.handler(payload)


class ToolRegistry:
    """Registry of callable tools that serializes to MCP schema."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add `tool` to the registry (overwrites on name collision)."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Fetch a tool by name."""
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def list(self) -> List[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_mcp_schema(self) -> List[Dict[str, Any]]:
        """Return the MCP-compatible list-of-tools schema."""
        return [t.to_mcp() for t in self._tools.values()]

    def to_json(self, indent: int = 2) -> str:
        """Return the MCP schema as a JSON string."""
        return json.dumps(self.to_mcp_schema(), indent=indent, sort_keys=True)


# ------------------------------------------------------------- python sandbox
_PY_SANDBOX_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python source to execute."},
        "timeout_sec": {
            "type": "number",
            "description": "Wall-clock timeout in seconds.",
            "default": 5.0,
            "minimum": 0.1,
            "maximum": 60.0,
        },
        "stdin": {
            "type": "string",
            "description": "Optional stdin to pipe into the child.",
            "default": "",
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


def _set_rlimits(cpu_sec: int, mem_bytes: int) -> None:
    """Apply best-effort POSIX resource limits in the child process.

    This is a mitigation, not an isolation boundary. A subprocess cannot be
    trusted to contain arbitrary untrusted code; use a dedicated sandbox
    (seccomp, nsjail, firecracker, etc.) for that.
    """
    if not _HAS_RESOURCE:  # pragma: no cover - Windows
        return
    try:
        _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
    except Exception:
        pass
    try:
        _resource.setrlimit(_resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass
    try:
        _resource.setrlimit(_resource.RLIMIT_NPROC, (64, 64))
    except Exception:
        pass


def _python_sandbox_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Python source in an isolated tempdir with a timeout.

    Security note: This is a convenience sandbox, NOT a security boundary.
    It only provides (1) wall-clock timeout, (2) tempdir working directory,
    and (3) best-effort POSIX rlimits (CPU, address space, process count).
    A determined attacker can still read many files, make network calls, or
    exhaust shared resources. Do not run untrusted code in production without
    a real isolation layer (nsjail, firecracker, gVisor, etc.).
    """
    code = payload.get("code")
    if not isinstance(code, str) or not code:
        raise ValueError("'code' must be a non-empty string")
    timeout = float(payload.get("timeout_sec", 5.0))
    stdin_data = str(payload.get("stdin", ""))

    with tempfile.TemporaryDirectory(prefix="hlf_sbx_") as td:
        tdp = Path(td)
        script = tdp / "run.py"
        script.write_text(code, encoding="utf-8")

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tdp),
            "TMPDIR": str(tdp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }

        preexec = None
        if _HAS_RESOURCE:
            def _pre() -> None:
                # 256 MiB, CPU ~= timeout+1s.
                _set_rlimits(int(timeout) + 1, 256 * 1024 * 1024)

            preexec = _pre

        started = time.time()
        timed_out = False
        stdout = ""
        stderr = ""
        rc: Optional[int] = None
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                cwd=str(tdp),
                env=env,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec,  # type: ignore[arg-type]
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            rc = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, (bytes, bytearray)) else (exc.stdout or "")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, (bytes, bytearray)) else (exc.stderr or "")

        elapsed = time.time() - started
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": rc,
            "timed_out": timed_out,
            "elapsed_sec": elapsed,
        }


python_sandbox_tool: Tool = Tool(
    name="python_sandbox",
    description=(
        "Run a short Python script inside a disposable temp directory with a "
        "wall-clock timeout and best-effort resource limits. Not a security "
        "boundary; do not execute untrusted code."
    ),
    json_schema=_PY_SANDBOX_SCHEMA,
    handler=_python_sandbox_handler,
)


# ----------------------------------------------------------------- filesystem
_FS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": ["read", "write", "list"],
            "description": "Filesystem operation to perform.",
        },
        "path": {
            "type": "string",
            "description": "Path relative to the allowed root.",
        },
        "content": {
            "type": "string",
            "description": "Text to write for op=='write'.",
            "default": "",
        },
    },
    "required": ["op", "path"],
    "additionalProperties": False,
}


def make_file_system_tool(allowed_root: os.PathLike) -> Tool:
    """Build a file-system tool rooted at `allowed_root`.

    Every path supplied by the caller is resolved and rejected if it escapes
    the allowed root via `..` or symlinks.

    Args:
        allowed_root: Directory that bounds all I/O.

    Returns:
        A `Tool` supporting `read`, `write`, and `list` operations.
    """
    root = Path(allowed_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def _resolve(rel: str) -> Path:
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError(
                f"path {rel!r} escapes allowed root {root}"
            ) from exc
        return candidate

    def _handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        op = payload.get("op")
        rel = payload.get("path")
        if not isinstance(op, str) or not isinstance(rel, str):
            raise ValueError("'op' and 'path' must be strings")

        if op == "read":
            target = _resolve(rel)
            if not target.is_file():
                raise FileNotFoundError(str(target))
            return {"content": target.read_text(encoding="utf-8")}
        if op == "write":
            target = _resolve(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = str(payload.get("content", ""))
            target.write_text(content, encoding="utf-8")
            return {"bytes_written": len(content.encode("utf-8"))}
        if op == "list":
            target = _resolve(rel)
            if not target.is_dir():
                raise NotADirectoryError(str(target))
            entries = []
            for child in sorted(target.iterdir()):
                entries.append(
                    {
                        "name": child.name,
                        "type": "dir" if child.is_dir() else "file",
                        "size": child.stat().st_size if child.is_file() else 0,
                    }
                )
            return {"entries": entries}
        raise ValueError(f"unknown op: {op}")

    return Tool(
        name="file_system",
        description=(
            f"Read/write/list files within the allowed root {root}. "
            "Path-traversal outside the root is rejected."
        ),
        json_schema=_FS_SCHEMA,
        handler=_handler,
    )


def build_default_registry(fs_root: Optional[os.PathLike] = None) -> ToolRegistry:
    """Construct a registry pre-populated with the two reference tools.

    Args:
        fs_root: Optional root directory for the file_system tool; a fresh
            temp directory is created when `None`.

    Returns:
        A populated `ToolRegistry`.
    """
    if fs_root is None:
        fs_root = tempfile.mkdtemp(prefix="hlf_fs_")
    registry = ToolRegistry()
    registry.register(python_sandbox_tool)
    registry.register(make_file_system_tool(fs_root))
    return registry


if __name__ == "__main__":
    reg = build_default_registry()
    print(reg.to_json())
