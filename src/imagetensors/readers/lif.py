"""Leica LIF format reader."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from readlif.reader import LifImage

import numpy as np
from readlif.reader import LifFile

from imagetensors.base import BaseImageReader
from imagetensors.models import ImageData, Metadata


class LifImageReader(BaseImageReader):
    """Reader for Leica LIF files.

    LIF files contain multiple series (images) in a single file.
    Each series is yielded as a separate ImageData object.
    """

    def __init__(self, image_path: str, override_pixel_size_um: float | None = None) -> None:
        super().__init__(image_path, override_pixel_size_um)
        self._lif_file = LifFile(str(self.path))

    def read(self) -> Iterator[ImageData]:
        """Read all series from the LIF file."""
        for series_idx in range(self._lif_file.num_images):
            lif_image = self._lif_file.get_image(img_n=series_idx)
            info = lif_image.info

            # Extract metadata
            metadata = Metadata(
                image_name=f'{self.path.stem}_Series_{series_idx + 1}.tif',
                source_path=self.path,
                series_index=series_idx,
                x_size=info['dims_n'].get(1, 0),
                y_size=info['dims_n'].get(2, 0),
                slices=info['dims_n'].get(3, 0),
                channels=info['channels'],
                frames=info['dims_n'].get(4, 0),
                x_resolution=info['scale_n'].get(1, 0),
                y_resolution=info['scale_n'].get(2, 0),
                time_dim=info['scale_n'].get(4, 0),
                begin=float(info['settings'].get('Begin', 0)) * 1e6,
                end=float(info['settings'].get('End', 0)) * 1e6,
            )

            # Build image array
            array = LifImageReader._build_array(lif_image, metadata)

            # Update display ranges
            ranges = self._calculate_ranges(array)
            metadata.Ranges = ranges['Ranges']
            metadata.min = ranges['min']
            metadata.max = ranges['max']

            # Build info string
            config = dict(info['settings'])
            metadata.Info = self._build_info_string(array, config)

            yield ImageData(array=array, metadata=metadata)

    @staticmethod
    def _build_array(lif_image: LifImage, metadata: Metadata) -> np.ndarray:
        """Build 5D array from LIF image."""
        # Collect all frames
        channels_data = []
        for c in range(metadata.channels):
            slices_data = []
            for z in range(metadata.slices):
                frame = np.array(lif_image.get_frame(z=z, c=c), dtype=np.uint8)
                slices_data.append(frame)

            # Stack slices: Z x Y x X
            channel_stack = np.stack(slices_data, axis=0)
            channels_data.append(channel_stack)

        # Stack channels: C x Z x Y x X
        array = np.stack(channels_data, axis=0)

        # Add time dimension: T x C x Z x Y x X
        array = np.expand_dims(array, axis=0)

        # Transpose to TZCYX
        array = np.transpose(array, (0, 2, 1, 3, 4))

        return array
