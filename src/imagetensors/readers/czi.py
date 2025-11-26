"""Zeiss CZI format reader."""

from collections.abc import Iterator
from typing import cast

import numpy as np
from czifile import CziFile

from imagetensors.base import BaseImageReader
from imagetensors.models import ImageData, Metadata

from ._czi_metadata import get_czi_metadata


class CziImageReader(BaseImageReader):
    """Reader for Zeiss CZI files.

    CZI files may contain phase-separated data (e.g., confocal vs AiryScan).
    Each phase is yielded as a separate ImageData object.

    Uses czitools for detailed metadata extraction if available,
    falls back to basic extraction otherwise.
    """

    def __init__(self, image_path: str, override_pixel_size_um: float | None = None) -> None:
        super().__init__(image_path, override_pixel_size_um)
        self._czi_file = CziFile(str(self.path))
        self._dimension_map = self._map_dimensions()

    def _map_dimensions(self) -> dict[str, int]:
        """Map CZI axes to their sizes."""
        axes = cast(list[str], self._czi_file.axes)
        shape = cast(list[int], self._czi_file.shape)
        dimension_map: dict[str, int] = dict(zip(axes, shape, strict=True))
        return dimension_map

    def read(self) -> Iterator[ImageData]:
        """Read all phases from the CZI file."""

        # ruff: noqa: PLR0914
        tensor = self._czi_file.asarray()

        # Remove singleton dimensions except standard ones
        for dim in list(self._dimension_map.keys()):
            if self._dimension_map[dim] == 1 and dim not in {'H', 'T', 'C', 'Z', 'Y', 'X'}:
                axis = list(self._dimension_map.keys()).index(dim)
                tensor = np.squeeze(tensor, axis=axis)
                del self._dimension_map[dim]

        # Handle phase dimension (H)
        if 'H' in self._dimension_map:
            position = list(self._dimension_map.keys()).index('H')
            tensor = np.flip(tensor, axis=position)

            num_phases = self._dimension_map['H']
            num_channels = self._dimension_map['C']
            channels_per_phase = num_channels // num_phases

            phase_tensors = []
            for phase_idx in range(num_phases):
                phase_channels = []

                for channel_offset in range(channels_per_phase):
                    channel_idx = phase_idx + channel_offset * num_phases
                    phase_tensor = tensor[phase_idx, :, channel_idx, :, :, :]
                    phase_channels.append(phase_tensor)

                # Stack: T x C x Z x Y x X
                phase_stack = np.stack(phase_channels, axis=1)
                # Transpose to: T x Z x C x Y x X
                phase_stack = np.transpose(phase_stack, (0, 2, 1, 3, 4))
                phase_tensors.append(phase_stack)
        else:
            # No phase separation: T x C x Z x Y x X -> T x Z x C x Y x X
            phase_tensors = [tensor[:, :, :, :, :]]
            num_phases = 1

        # Yield each phase
        mode_names = {0: 'Confocal', 1: 'AiryScan'}

        for phase_idx, phase_tensor in enumerate(phase_tensors):
            # Get metadata for this phase
            metadata_dict = get_czi_metadata(
                str(self.path),
                phase_index=phase_idx,
                num_phases=num_phases,
            )

            # Determine image name
            if num_phases == 2:
                image_name = f'{self.path.stem}_{mode_names.get(phase_idx, phase_idx)}.tif'
            else:
                image_name = f'{self.path.stem}_Phase_{phase_idx + 1}.tif'

            # Build metadata
            channels_total = metadata_dict['Dimensions'].get('SizeC', 0)
            channels_per_phase = channels_total // num_phases if num_phases > 1 else channels_total

            metadata = Metadata(
                image_name=image_name,
                source_path=self.path,
                series_index=phase_idx,
                x_size=metadata_dict['Dimensions'].get('SizeX', 0),
                y_size=metadata_dict['Dimensions'].get('SizeY', 0),
                slices=metadata_dict['Dimensions'].get('SizeZ', 0),
                channels=channels_per_phase,
                frames=metadata_dict['Dimensions'].get('SizeT', 0),
                x_resolution=1.0 / metadata_dict['Scaling'].get('X', 1),
                y_resolution=1.0 / metadata_dict['Scaling'].get('Y', 1),
                time_dim=metadata_dict['Scaling'].get('T', 1),
                begin=0.0,
                end=float(metadata_dict['Scaling'].get('Z', 0) * (metadata_dict['Dimensions'].get('SizeZ', 0) - 1)),
            )

            # Calculate ranges
            ranges = self._calculate_ranges(phase_tensor)
            metadata.Ranges = ranges['Ranges']
            metadata.min = ranges['min']
            metadata.max = ranges['max']

            # Build info string with full metadata
            metadata.Info = self._build_info_string(phase_tensor, metadata_dict)

            yield ImageData(array=phase_tensor, metadata=metadata)
