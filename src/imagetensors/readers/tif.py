"""TIFF format reader."""

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import tifffile

from imagetensors.base import BaseImageReader
from imagetensors.models import ImageData, Metadata


class TifImageReader(BaseImageReader):
    """Reader for TIFF stack files.

    Supports ImageJ-style TIFF with metadata.
    """

    def __init__(self, image_path: str, override_pixel_size_um: float | None = None) -> None:
        super().__init__(image_path, override_pixel_size_um)

    def read(self) -> Iterator[ImageData]:
        """Read the TIFF file."""
        with tifffile.TiffFile(str(self.path)) as tif:
            # Load array
            array = tif.asarray()

            # Get axes from series
            axes = tif.series[0].axes

            # Get metadata
            imagej_metadata = tif.imagej_metadata or {}

            # Get resolution
            x_res = TifImageReader._get_resolution(tif, 282)  # XResolution
            y_res = TifImageReader._get_resolution(tif, 283)  # YResolution

            # Expand to 5D (TZCYX)
            array = TifImageReader._expand_to_5d(array, axes)

            # Build metadata
            spacing = imagej_metadata.get('spacing', 0.0)

            metadata = Metadata(
                image_name=self.path.name,
                source_path=self.path,
                series_index=0,
                x_size=array.shape[4],
                y_size=array.shape[3],
                slices=array.shape[1],
                channels=array.shape[2],
                frames=array.shape[0],
                x_resolution=x_res,
                y_resolution=y_res,
                time_dim=1.0,
                begin=0.0,
                end=float((array.shape[1] - 1) * spacing),
            )

            # Calculate ranges
            ranges = self._calculate_ranges(array)
            metadata.Ranges = ranges['Ranges']
            metadata.min = ranges['min']
            metadata.max = ranges['max']

            # Extract configuration from Info string
            config = {}
            if 'Info' in imagej_metadata:
                config = TifImageReader._parse_info_string(imagej_metadata['Info'])

            # Build info string
            metadata.Info = self._build_info_string(array, config)

            yield ImageData(array=array, metadata=metadata)

    @staticmethod
    def _get_resolution(tif: tifffile.TiffFile, tag_code: int) -> float:
        """Extract resolution from TIFF tags."""
        first_page = cast(Any, tif.pages[0])

        if tag_code not in first_page.tags:
            return 1.0

        res_value = first_page.tags[tag_code].value

        if isinstance(res_value, tuple) and len(res_value) == 2:
            numerator, denominator = res_value
            return float(numerator) / float(denominator)

        return float(res_value)

    @staticmethod
    def _expand_to_5d(array: np.ndarray, axes: str) -> np.ndarray:
        """Expand array to 5D TZCYX format."""
        all_axes = 'TZCYX'

        # Find missing axes
        missing_axes = [i for i, dim in enumerate(all_axes) if dim not in axes]

        # Add singleton dimensions
        for missing_idx in missing_axes:
            array = np.expand_dims(array, axis=missing_idx)

        return array

    @staticmethod
    def _parse_info_string(info_str: str) -> dict[str, str]:
        """Parse ImageJ Info string into configuration dict."""
        config = {}

        for line in info_str.splitlines():
            if '[' not in line or '=' not in line:
                continue

            try:
                key_part, value_part = line.split(' = ', 1)
                key = key_part.strip('[] ')
                value = value_part.strip()
                config[key] = value
            except ValueError:
                continue

        return config
