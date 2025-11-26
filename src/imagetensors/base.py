"""Base reader interface and utilities."""

import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .models import ImageData


class BaseImageReader(ABC):
    """Abstract base class for microscopy image readers.

    Each reader handles a specific file format and yields ImageData objects
    with standardized 5D arrays (TZCYX) and metadata.
    """

    def __init__(self, image_path: str, override_pixel_size_um: float | None = None) -> None:
        """Initialize reader.

        Args:
            image_path: Path to the image file
            override_pixel_size_um: Optional manual pixel size override
        """
        self.path = Path(image_path).resolve()
        self._override_pixel_size_um = override_pixel_size_um

        if not self.path.exists():
            raise FileNotFoundError(f'Image file not found: {self.path}')

    @abstractmethod
    def read(self) -> Iterator[ImageData]:
        """Read and yield ImageData objects.

        Yields:
            ImageData objects with 5D arrays and metadata
        """

    def __iter__(self) -> Iterator[ImageData]:
        """Allow iteration over reader."""
        return self.read()

    def _build_info_string(self, array: 'np.ndarray', config: dict[str, str] | None = None) -> str:
        """Build ImageJ-compatible info string.

        Args:
            array: The image array
            config: Optional configuration dictionary

        Returns:
            Formatted info string
        """
        t, z, c, y, x = array.shape
        bits = array.dtype.itemsize * 8
        pixel_type = str(array.dtype)

        # Determine byte order
        byte_order = array.dtype.byteorder
        if byte_order in {'=', '|'}:
            little_endian = sys.byteorder == 'little'
        elif byte_order == '<':
            little_endian = True
        elif byte_order == '>':
            little_endian = False
        else:
            little_endian = True  # default

        info = (
            f' BitsPerPixel = {bits}\r\n'
            f' DimensionOrder = TZCYX\r\n'
            f' IsInterleaved = false\r\n'
            f' LittleEndian = {str(little_endian).lower()}\r\n'
            f' PixelType = {pixel_type}\r\n'
            f' SizeC = {c}\r\n'
            f' SizeT = {t}\r\n'
            f' SizeX = {x}\r\n'
            f' SizeY = {y}\r\n'
            f' SizeZ = {z}\r\n'
        )

        # Add configuration if provided
        if config:
            config_lines = self._flatten_config(config)
            info += config_lines

        return info

    def _flatten_config(self, config: dict[str, str], parent_keys: list[str] | None = None) -> str:
        """Flatten nested configuration dictionary to string.

        Args:
            config: Configuration dictionary
            parent_keys: Parent key path for recursion

        Returns:
            Flattened configuration string
        """
        if parent_keys is None:
            parent_keys = []

        lines = []
        for key, value in sorted(config.items()):
            current_path = [*parent_keys, key]

            if isinstance(value, dict):
                lines.append(self._flatten_config(value, current_path))
            elif value is not None:
                key_path = ''.join(f'[{k}]' for k in current_path)
                lines.append(f'{key_path} = {value}\r\n')

        return ''.join(lines)

    def _calculate_ranges(self, array: np.ndarray) -> dict[str, Any]:
        """Calculate display ranges for ImageJ."""
        _ = self

        ranges = []
        for c in range(array.shape[2]):  # channels
            channel_data = array[:, :, c, :, :]
            ranges.append(float(channel_data.min()))
            ranges.append(float(channel_data.max()))

        return {
            'Ranges': tuple(ranges),
            'min': float(array.min()),
            'max': float(array.max()),
        }
