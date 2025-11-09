"""Factory for creating appropriate image readers."""

from pathlib import Path
from typing import Iterator

from .models import ImageData
from .readers import (
    CziImageReader,
    LifImageReader,
    NdImageReader,
    OibImageReader,
    TifImageReader,
)


class ImageReader:
    """Factory class for reading microscopy images.
    
    Automatically selects the appropriate reader based on file extension.
    
    Example:
        >>> reader = ImageReader("path/to/image.lif")
        >>> for image_data in reader:
        >>>     process(image_data.array)
    """
    
    READERS = {
        '.lif': LifImageReader,
        '.czi': CziImageReader,
        '.oib': OibImageReader,
        '.nd': NdImageReader,
        '.tif': TifImageReader,
        '.tiff': TifImageReader,
    }
    
    def __init__(self, file_path: str, override_pixel_size_um: float = None):
        """Initialize reader for the given file.
        
        Args:
            file_path: Path to microscopy image file
            override_pixel_size_um: Optional manual pixel size override
            
        Raises:
            ValueError: If file format is not supported
        """
        self.path = Path(file_path)
        extension = self.path.suffix.lower()
        
        if extension not in self.READERS:
            supported = ', '.join(self.READERS.keys())
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {supported}"
            )
        
        reader_class = self.READERS[extension]
        self._reader = reader_class(file_path, override_pixel_size_um)
    
    def __iter__(self) -> Iterator[ImageData]:
        """Iterate over all images in the file."""
        return iter(self._reader)
    
    def read(self) -> Iterator[ImageData]:
        """Read all images from the file.
        
        Yields:
            ImageData objects with 5D arrays and metadata
        """
        return self._reader.read()