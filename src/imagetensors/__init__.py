"""Microscopy image reading and conversion for BioPixel.

This module provides unified access to multiple proprietary microscopy formats,
converting them to standardized 5D arrays (TZCYX) for downstream analysis.

Supported formats:
- Leica LIF (multi-series)
- Zeiss CZI (with phase separation)
- Olympus OIB
- Nikon ND (multi-file assemblies)
- TIFF stacks

Example:
    >>> from imagetensors import ImageReader
    >>> 
    >>> reader = ImageReader("path/to/image.lif")
    >>> for image_data in reader:
    >>>     # image_data.array is 5D numpy array (TZCYX)
    >>>     # image_data.metadata contains all metadata
    >>>     process(image_data.array)
"""

from .factory import ImageReader
from .models import ImageData, Metadata
from .converters import save_as_tif

__all__ = ["ImageReader", "ImageData", "Metadata", "save_as_tif"]