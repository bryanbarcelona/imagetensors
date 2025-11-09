"""TIFF format reader."""

from typing import Iterator

import numpy as np
import tifffile

from ..models import ImageData, Metadata
from ..base import BaseImageReader


class TifImageReader(BaseImageReader):
    """Reader for TIFF stack files.
    
    Supports ImageJ-style TIFF with metadata.
    """
    
    def __init__(self, image_path: str, override_pixel_size_um: float = None):
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
            x_res = self._get_resolution(tif, 282)  # XResolution
            y_res = self._get_resolution(tif, 283)  # YResolution
            
            # Expand to 5D (TZCYX)
            array = self._expand_to_5d(array, axes)
            
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
                config = self._parse_info_string(imagej_metadata['Info'])
            
            # Build info string
            metadata.Info = self._build_info_string(array, config)
            
            yield ImageData(array=array, metadata=metadata)
    
    def _get_resolution(self, tif: tifffile.TiffFile, tag_code: int) -> float:
        """Extract resolution from TIFF tags."""
        if tag_code not in tif.pages[0].tags:
            return 1.0
        
        res_value = tif.pages[0].tags[tag_code].value
        
        if isinstance(res_value, tuple) and len(res_value) == 2:
            numerator, denominator = res_value
            return float(numerator) / float(denominator)
        
        return float(res_value)
    
    def _expand_to_5d(self, array: np.ndarray, axes: str) -> np.ndarray:
        """Expand array to 5D TZCYX format."""
        all_axes = 'TZCYX'
        
        # Find missing axes
        missing_axes = [i for i, dim in enumerate(all_axes) if dim not in axes]
        
        # Add singleton dimensions
        for missing_idx in missing_axes:
            array = np.expand_dims(array, axis=missing_idx)
        
        return array
    
    def _parse_info_string(self, info_str: str) -> dict:
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
    
    def _calculate_ranges(self, array: np.ndarray) -> dict:
        """Calculate display ranges for ImageJ."""
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