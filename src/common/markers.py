"""Centralized AprilTag marker configuration and registry loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence
import yaml

log = logging.getLogger("apriltag-markers")

# Default in-code fallback mappings (134mm for 0, 41mm for 20-31, 66mm for 40-45)
DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE_M = 0.0885

FALLBACK_MARKER_SIZES_M: dict[int, float] = {
    0: 0.1340,
    1: 0.1240,
    **{i: 0.0885 for i in range(2, 6)},
    **{i: 0.0575 for i in range(6, 14)},
    **{i: 0.0410 for i in range(20, 32)},
    **{i: 0.0660 for i in range(40, 46)},
}


def _find_markers_yaml(custom_path: Path | str | None = None) -> Path | None:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        Path.cwd() / "calibrations" / "markers.yaml",
        repo_root / "calibrations" / "markers.yaml",
        Path("/home/mfk01/Projects/econ-systems/demos/a2-01/calibrations/markers.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _parse_id_range(key_str: str | int) -> list[int]:
    if isinstance(key_str, int):
        return [key_str]
    s = str(key_str).strip()
    if ".." in s:
        parts = s.split("..", 1)
        try:
            start, end = int(parts[0].strip()), int(parts[1].strip())
            return list(range(start, end + 1))
        except ValueError:
            return []
    if "-" in s and not s.startswith("-"):
        parts = s.split("-", 1)
        try:
            start, end = int(parts[0].strip()), int(parts[1].strip())
            return list(range(start, end + 1))
        except ValueError:
            return []
    try:
        return [int(s)]
    except ValueError:
        return []


class MarkerRegistry:
    """Manages AprilTag marker sizes, families, and metadata."""

    def __init__(self, config_path: Path | str | None = None):
        self.config_file = _find_markers_yaml(config_path)
        self.default_family: str = DEFAULT_TAG_FAMILY
        self.default_size_m: float = DEFAULT_TAG_SIZE_M
        self.sizes_m: dict[int, float] = dict(FALLBACK_MARKER_SIZES_M)
        self.metadata: dict[int, dict[str, str | float]] = {}
        self.reload()

    def reload(self):
        if not self.config_file or not self.config_file.exists():
            log.debug("No markers.yaml found; using internal defaults.")
            self._build_metadata()
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self.default_family = str(data.get("default_family", DEFAULT_TAG_FAMILY))
            self.default_size_m = float(data.get("default_size_m", DEFAULT_TAG_SIZE_M))

            raw_markers = data.get("markers", {})
            for key, val in raw_markers.items():
                ids = _parse_id_range(key)
                if not ids:
                    continue

                if isinstance(val, (int, float)):
                    size_m = float(val)
                    fam = self.default_family
                    desc = ""
                elif isinstance(val, dict):
                    size_m = float(val.get("size_m", self.default_size_m))
                    fam = str(val.get("family", self.default_family))
                    desc = str(val.get("description", ""))
                else:
                    continue

                for tag_id in ids:
                    self.sizes_m[tag_id] = size_m
                    self.metadata[tag_id] = {
                        "family": fam,
                        "size_m": size_m,
                        "size_mm": round(size_m * 1000.0, 1),
                        "description": desc,
                    }

        except Exception as e:
            log.warning(f"Failed to parse {self.config_file}: {e}. Retaining internal defaults.")

        self._build_metadata()

    def _build_metadata(self):
        for tag_id, size_m in self.sizes_m.items():
            if tag_id not in self.metadata:
                self.metadata[tag_id] = {
                    "family": self.default_family,
                    "size_m": size_m,
                    "size_mm": round(size_m * 1000.0, 1),
                    "description": "",
                }

    def get_size(self, tag_id: int, fallback: float | None = None) -> float:
        if fallback is None:
            fallback = self.default_size_m
        return self.sizes_m.get(tag_id, fallback)

    def get_family(self, tag_id: int | None = None) -> str:
        if tag_id is not None and tag_id in self.metadata:
            return str(self.metadata[tag_id].get("family", self.default_family))
        return self.default_family

    def get_sizes_dict(self) -> dict[int, float]:
        return dict(self.sizes_m)


# Global singleton instance
_registry = MarkerRegistry()


def get_marker_sizes() -> dict[int, float]:
    """Return dictionary mapping tag ID -> physical size in meters."""
    return _registry.get_sizes_dict()


def get_marker_size(tag_id: int, fallback: float | None = None) -> float:
    """Return physical size in meters for a given tag ID."""
    return _registry.get_size(tag_id, fallback=fallback)


def get_tag_family(tag_id: int | None = None) -> str:
    """Return AprilTag family string (default: tag36h11)."""
    return _registry.get_family(tag_id)


def get_marker_metadata(tag_id: int | None = None) -> dict:
    """Return metadata dict for a specific tag or all tags."""
    if tag_id is not None:
        return _registry.metadata.get(tag_id, {
            "family": _registry.default_family,
            "size_m": _registry.default_size_m,
            "size_mm": round(_registry.default_size_m * 1000.0, 1),
            "description": "",
        })
    return dict(_registry.metadata)


def parse_marker_size_overrides(entries: Sequence[str] | None, base_sizes: dict[int, float] | None = None) -> dict[int, float]:
    """Parse CLI override list (e.g. ['0=0.134', '1=0.124']) on top of registered sizes."""
    sizes = dict(base_sizes if base_sizes is not None else get_marker_sizes())
    if not entries:
        return sizes
    for entry in entries:
        if "=" in entry:
            k, v = entry.split("=", 1)
            try:
                sizes[int(k.strip())] = float(v.strip())
            except ValueError:
                log.warning(f"Invalid marker size entry: {entry}")
    return sizes


DEFAULT_MARKER_SIZES_M = get_marker_sizes()
DEFAULT_MARKER_METADATA = get_marker_metadata()
