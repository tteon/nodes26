"""Read the approved project .env without executing shell or exposing values."""

import os
from pathlib import Path
import re
import shlex


RUNTIME_KEYS = frozenset(
    {
        "MARA_API_KEY",
        "MARA_BASE_URL",
        "MARA_ADMISSION_LOCK",
        "DOZER_URI",
        "DOZER_USER",
        "DOZER_PASSWORD",
    }
)


def load_project_environment(path=None):
    path = (
        Path(path) if path is not None else Path(__file__).resolve().parents[4] / ".env"
    )
    if not path.is_file():
        return {"file_present": False, "loaded_names": []}
    parsed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=(.*)$", line)
        if not match or match[1] not in RUNTIME_KEYS:
            continue
        try:
            parts = shlex.split(match[2], comments=True, posix=True)
            if len(parts) > 1:
                raise ValueError()
        except ValueError:
            raise ValueError("Invalid .env syntax for " + match[1]) from None
        parsed[match[1]] = parts[0] if parts else ""
    loaded = []
    for name, value in parsed.items():
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return {"file_present": True, "loaded_names": sorted(loaded)}
