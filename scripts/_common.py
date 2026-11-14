"""Shared start-up for the numbered pipeline scripts: environment variables and logging.

Global context: each script in scripts/ is a thin wrapper around one pipeline phase implemented in
src/racingmania; this module only prepares the process (Triton floating-point fusion setting,
timestamped logging) so every script behaves the same.
"""
import os
import sys

# Windows consoles default to cp1252; libraries (e.g. the ONNX exporter) print emoji. Force UTF-8 output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # keep Triton's current default explicitly (silences a warning)

from racingmania.utils.logging import setup_logging  # noqa: E402

setup_logging()
