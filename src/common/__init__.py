"""Common utilities and registries for multi-camera and realsense demos."""

from common.markers import (
    DEFAULT_MARKER_METADATA,
    DEFAULT_MARKER_SIZES_M,
    DEFAULT_TAG_FAMILY,
    DEFAULT_TAG_SIZE_M,
    MarkerRegistry,
    get_marker_metadata,
    get_marker_size,
    get_marker_sizes,
    get_tag_family,
    parse_marker_size_overrides,
)

__all__ = [
    "DEFAULT_TAG_FAMILY",
    "DEFAULT_TAG_SIZE_M",
    "DEFAULT_MARKER_SIZES_M",
    "DEFAULT_MARKER_METADATA",
    "MarkerRegistry",
    "get_marker_sizes",
    "get_marker_size",
    "get_tag_family",
    "get_marker_metadata",
    "parse_marker_size_overrides",
]
