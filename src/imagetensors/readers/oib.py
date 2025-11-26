"""Olympus OIB format reader."""

from collections.abc import Iterator

import numpy as np
from oiffile import OifFile

from imagetensors.base import BaseImageReader
from imagetensors.models import ImageData, Metadata


class OibImageReader(BaseImageReader):
    """Reader for Olympus OIB files.

    OIB files typically contain a single image.
    """

    def __init__(self, image_path: str, override_pixel_size_um: float | None = None) -> None:
        super().__init__(image_path, override_pixel_size_um)
        self._oib_file = OifFile(str(self.path))

    def read(self) -> Iterator[ImageData]:
        """Read the OIB file."""
        mainfile = self._oib_file.mainfile

        # Extract axis information
        axis_info = {
            mainfile['Axis 0 Parameters Common']['AxisCode']: mainfile['Axis 0 Parameters Common']['MaxSize'],
            mainfile['Axis 1 Parameters Common']['AxisCode']: mainfile['Axis 1 Parameters Common']['MaxSize'],
            mainfile['Axis 2 Parameters Common']['AxisCode']: mainfile['Axis 2 Parameters Common']['MaxSize'],
            mainfile['Axis 3 Parameters Common']['AxisCode']: mainfile['Axis 3 Parameters Common']['MaxSize'],
            mainfile['Axis 4 Parameters Common']['AxisCode']: mainfile['Axis 4 Parameters Common']['MaxSize'],
        }

        x_conv = round(mainfile['Reference Image Parameter']['WidthConvertValue'], 4)
        y_conv = round(mainfile['Reference Image Parameter']['HeightConvertValue'], 4)
        z_start = mainfile['Axis 3 Parameters Common']['StartPosition']
        z_end = mainfile['Axis 3 Parameters Common']['EndPosition']

        # Build metadata
        metadata = Metadata(
            image_name=f'{self.path.stem}.tif',
            source_path=self.path,
            series_index=0,
            x_size=axis_info.get('X', 1),
            y_size=axis_info.get('Y', 1),
            slices=axis_info.get('Z', 1),
            channels=axis_info.get('C', 1),
            frames=axis_info.get('T', 1),
            x_resolution=1.0 / x_conv,
            y_resolution=1.0 / y_conv,
            time_dim=1.0,
            begin=z_start,
            end=z_end,
        )

        # Load array
        array = self._oib_file.asarray()

        # Add missing dimensions
        for dim_idx, dim_code in enumerate('CZTYX'):
            target = axis_info.get(dim_code, 1)
            if array.shape[dim_idx] != target and target != 0:
                axis_info[dim_code] = 1
                array = np.expand_dims(array, axis=dim_idx)

        # Transpose to TZCYX and reverse channel order
        array = array.transpose(2, 1, 0, 3, 4)  # TZCYX
        array = array[:, :, ::-1, :, :]  # Reverse channels (high to low wavelength)

        # Calculate ranges
        ranges = self._calculate_ranges(array)
        metadata.Ranges = ranges['Ranges']
        metadata.min = ranges['min']
        metadata.max = ranges['max']

        # Build info string
        config = dict(mainfile)
        metadata.Info = self._build_info_string(array, config)

        yield ImageData(array=array, metadata=metadata)

    def __del__(self) -> None:
        """Clean up file handle."""
        if hasattr(self, '_oib_file') and self._oib_file:
            self._oib_file.close()
