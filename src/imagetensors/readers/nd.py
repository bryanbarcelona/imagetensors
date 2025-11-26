"""Nikon ND format reader."""

import pathlib
import re
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import tifffile

from imagetensors.base import BaseImageReader
from imagetensors.models import ImageData, Metadata


class NdImageReader(BaseImageReader):
    """Reader for Nikon ND files.

    ND files are manifests that reference multiple STK/TIF files.
    Files are assembled into one or more 5D arrays based on stage positions.
    """

    def __init__(self, image_path: str, override_pixel_size_um: float | None = None) -> None:
        super().__init__(image_path, override_pixel_size_um)
        self._nd_config: dict[str, Any] = {}
        self._stage_count: int = 1
        self._parse_nd_file()
        self._find_associated_files()
        self._extract_metadata_from_tif()
        self._build_file_tree()

    def _parse_nd_file(self) -> None:
        """Parse the ND manifest file."""
        with pathlib.Path(self.path).open(encoding='utf-8', errors='ignore') as f:
            for raw_line in f:
                line = raw_line.replace('"', '').strip()
                if not line:
                    continue

                parts = line.split(',')
                key = parts[0].strip()

                value = parts[1].strip() if len(parts) == 2 else '.'.join(parts[1:]).strip()

                # Type conversion
                result: str | int | float = value

                if value.isdigit():
                    result = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    result = float(value)

                self._nd_config[key] = result

        # Extract dimensions
        self._metadata = Metadata(source_path=self.path)

        if self._nd_config.get('DoTimelapse', 'FALSE').upper() == 'TRUE':
            self._metadata.frames = self._nd_config.get('NTimePoints', 1)
        else:
            self._metadata.frames = 1

        if self._nd_config.get('DoWave', 'FALSE').upper() == 'TRUE':
            self._metadata.channels = self._nd_config.get('NWavelengths', 1)
        else:
            self._metadata.channels = 1

        if self._nd_config.get('DoZSeries', 'FALSE').upper() == 'TRUE':
            self._metadata.slices = self._nd_config.get('NZSteps', 1)
            z_step = self._nd_config.get('ZStepSize', 0)
            self._metadata.end = z_step * (self._metadata.slices - 1)
        else:
            self._metadata.slices = 1
            self._metadata.end = 0

        if self._nd_config.get('DoStage', 'FALSE').upper() == 'TRUE':
            self._stage_count = self._nd_config.get('NStagePositions', 1)
        else:
            self._stage_count = 1

        self._metadata.begin = 0

    def _find_associated_files(self) -> None:
        """Find all files associated with this ND file."""
        base_name = self.path.stem
        parent_dir = self.path.parent

        pattern = f'{base_name}*'
        self._associated_files = [
            f for f in parent_dir.glob(pattern) if f.suffix.lower() in {'.tif', '.tiff', '.stk'} and f.is_file()
        ]

        if not self._associated_files:
            raise ValueError(f'No associated image files found for {self.path}')

    def _extract_metadata_from_tif(self) -> None:
        """Extract metadata from the first TIF file."""
        if not self._associated_files:
            return

        with tifffile.TiffFile(self._associated_files[0]) as tif:
            first_page = cast(Any, tif.pages[0])
            tags = first_page.tags

            self._metadata.x_size = int(tags[256].value)  # ImageWidth
            self._metadata.y_size = int(tags[257].value)  # ImageLength

            x_res = float(tags[282].value[0]) if 282 in tags else 1.0

            y_res = float(tags[283].value[0]) if 283 in tags else 1.0

            # Handle calibration from MM metadata
            if 33628 in tags:
                mm_metadata = tags[33628].value

                x_cal = mm_metadata.get('XCalibration', 0)
                y_cal = mm_metadata.get('YCalibration', 0)

                if self._override_pixel_size_um is not None:
                    self._metadata.x_resolution = 1.0 / self._override_pixel_size_um
                    self._metadata.y_resolution = 1.0 / self._override_pixel_size_um
                elif x_cal == 0:
                    self._metadata.x_resolution = 1.0 / (x_res / 10000)
                    self._metadata.y_resolution = 1.0 / (y_res / 10000)
                else:
                    self._metadata.x_resolution = 1.0 / x_cal
                    self._metadata.y_resolution = 1.0 / y_cal
            else:
                self._metadata.x_resolution = 1.0 / (x_res / 10000)
                self._metadata.y_resolution = 1.0 / (y_res / 10000)

    def _build_file_tree(self) -> None:
        """Organize files into a tree structure: stage -> time -> channel -> files."""
        # ruff: noqa: C901
        base_path = str(self.path.with_suffix(''))

        # Remove base path from filenames
        relative_files = [str(f).replace(base_path, '') for f in self._associated_files]

        # Parse file indices
        w_indices = set()
        s_indices = set()
        t_indices = set()

        for filename in relative_files:
            if match := re.search(r'_w(\d+)', filename):
                w_indices.add(match.group(1))
            if match := re.search(r'_s(\d+)', filename):
                s_indices.add(match.group(1))
            if match := re.search(r'_t(\d+)', filename):
                t_indices.add(match.group(1))

        num_w = max(len(w_indices), 1)
        num_s = max(len(s_indices), 1)
        num_t = max(len(t_indices), 1)

        # Validate counts
        if num_w != self._metadata.channels or num_s != self._stage_count or num_t != self._metadata.frames:
            raise ValueError(
                f'File count mismatch: expected {self._metadata.channels} channels, '
                f'{self._stage_count} stages, {self._metadata.frames} timepoints; '
                f'found {num_w}, {num_s}, {num_t}'
            )

        # Build tree
        self._file_tree = []
        for stage in range(self._stage_count):
            stage_files = [f for f in relative_files if f'_s{stage + 1}' in f or num_s == 1]
            if not stage_files:
                stage_files = relative_files.copy()

            stage_tree = []
            for time in range(self._metadata.frames):
                time_pattern = f'_t{time + 1}[_.]'
                time_files = [f for f in stage_files if re.search(time_pattern, f) or num_t == 1]
                if not time_files:
                    time_files = stage_files.copy()

                time_tree = []
                for channel in range(self._metadata.channels):
                    channel_files = [f for f in time_files if f'_w{channel + 1}' in f or num_w == 1]
                    if not channel_files:
                        channel_files = time_files.copy()

                    # Add back base path
                    channel_files = [base_path + f for f in channel_files]
                    time_tree.append(channel_files)

                stage_tree.append(time_tree)

            self._file_tree.append(stage_tree)

    def read(self) -> Iterator[ImageData]:
        """Read all stage positions as separate ImageData objects."""
        for stage_idx, stage_tree in enumerate(self._file_tree):
            # Build 5D array for this stage
            time_series = []

            for time_tree in stage_tree:
                channel_series = []

                for channel_files in time_tree:
                    # Load all Z slices for this channel
                    stacks = [tifffile.imread(f) for f in channel_files]

                    # Handle 2D images
                    stacks = [np.expand_dims(s, axis=0) if s.ndim == 2 else s for s in stacks]

                    # Stack: Z x Y x X
                    z_stack = np.concatenate(stacks, axis=0)
                    channel_series.append(z_stack)

                # Stack channels: C x Z x Y x X
                channel_stack = np.stack(channel_series, axis=0)
                time_series.append(channel_stack)

            # Stack time: T x C x Z x Y x X
            array = np.stack(time_series, axis=0)

            # Transpose to TZCYX
            array = np.transpose(array, (0, 2, 1, 3, 4))

            # Reverse channels (high to low wavelength)
            array = array[:, :, ::-1, :, :]

            # Build metadata for this stage
            metadata = Metadata(
                image_name=f'{self.path.stem}_Series_{stage_idx + 1}.tif',
                source_path=self.path,
                series_index=stage_idx,
                x_size=self._metadata.x_size,
                y_size=self._metadata.y_size,
                slices=self._metadata.slices,
                channels=self._metadata.channels,
                frames=self._metadata.frames,
                x_resolution=self._metadata.x_resolution,
                y_resolution=self._metadata.y_resolution,
                time_dim=1.0,
                begin=self._metadata.begin,
                end=self._metadata.end,
            )

            # Calculate ranges
            ranges = self._calculate_ranges(array)
            metadata.Ranges = ranges['Ranges']
            metadata.min = ranges['min']
            metadata.max = ranges['max']

            # Build info string
            metadata.Info = self._build_info_string(array, self._nd_config)

            yield ImageData(array=array, metadata=metadata)
