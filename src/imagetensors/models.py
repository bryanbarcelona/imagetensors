"""Data models for microscopy images."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np


@dataclass
class Metadata:
    """Metadata for a microscopy image.

    All spatial measurements in micrometers unless otherwise noted.
    """

    # Image identification
    image_name: str | None = None
    source_path: Path | None = None
    series_index: int | None = None

    # Dimensions
    x_size: int = 0
    y_size: int = 0
    slices: int = 0  # Z dimension
    channels: int = 0
    frames: int = 0  # Time dimension

    # Physical scaling (micrometers per pixel/voxel)
    x_resolution: float = 0.0  # µm/pixel
    y_resolution: float = 0.0  # µm/pixel
    time_dim: float = 0.0  # time per frame

    # Z-stack range
    begin: float = 0.0  # Z start position (µm)
    end: float = 0.0  # Z end position (µm)

    # Display ranges (for ImageJ compatibility)
    min: float = 0.0
    max: float = 0.0
    Ranges: tuple[float, float] | None = None

    # Additional metadata
    Info: str | None = None

    @property
    def z_range(self) -> float:
        """Total Z-stack range in micrometers."""
        return abs(self.end - self.begin)

    @property
    def spacing(self) -> float:
        """Z-spacing between slices in micrometers.

        Calculated from z_range and number of slices.
        Returns 0 if only one slice.
        """
        if self.slices <= 1:
            return 0.0
        return self.z_range / (self.slices - 1)

    def to_imagej_metadata(self) -> dict[str, object]:
        """Convert to ImageJ-compatible metadata dictionary."""
        return {
            'axes': 'TZCYX',
            'spacing': self.spacing,
            'unit': 'micron',
            'hyperstack': 'true',
            'mode': 'color',
            'channels': self.channels,
            'frames': self.frames,
            'slices': self.slices,
            'Info': self.Info or '',
            'Ranges': self.Ranges,
            'min': self.min,
            'max': self.max,
            'metadata': 'ImageJ=1.53c\n',
            'x_resolution': self.x_resolution,
            'y_resolution': self.y_resolution,
        }


@dataclass
class ImageData:
    """A microscopy image with its metadata.

    Represents a single 5D image in TZCYX format.
    """

    array: np.ndarray  # Shape: (T, Z, C, Y, X)
    metadata: Metadata

    def __post_init__(self) -> None:
        """Validate array dimensions."""
        if self.array.ndim != 5:
            raise ValueError(f'Image array must be 5D (TZCYX), got {self.array.ndim}D')

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """Array shape (T, Z, C, Y, X)."""
        return cast(tuple[int, int, int, int, int], self.array.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Array data type."""
        return self.array.dtype

    def __repr__(self) -> str:
        return f'ImageData(shape={self.shape}, dtype={self.dtype}, series={self.metadata.series_index})'
