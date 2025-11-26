"""Image conversion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tifffile

from .models import ImageData

if TYPE_CHECKING:
    from imagetensors import ImageReader


def save_as_tif(
    image_data: ImageData,
    output_path: str | Path,
    compression: str | None = None,
) -> Path:
    """Save ImageData as ImageJ-compatible TIFF.

    Args:
        image_data: ImageData object to save
        output_path: Output file path
        compression: Optional compression ('zlib', 'lzw', etc.)

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = image_data.metadata.to_imagej_metadata()

    tifffile.imwrite(
        output_path,
        image_data.array,
        imagej=True,
        metadata=metadata,
        compression=compression,
    )

    return output_path


def save_all_as_tif(
    reader: ImageReader,
    output_dir: str | Path,
    name_template: str | None = None,
    compression: str | None = None,
) -> list[Path]:
    """Save all images from a reader as TIFF files.

    Args:
        reader: ImageReader instance
        output_dir: Output directory
        name_template: Optional name template (e.g., "{stem}_series_{idx}.tif")
        compression: Optional compression

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for idx, image_data in enumerate(reader):
        if name_template:
            filename = name_template.format(
                stem=reader.path.stem,
                idx=idx + 1,
                series=image_data.metadata.series_index,
            )
        else:
            filename = image_data.metadata.image_name

        if filename is None:
            raise ValueError('Filename cannot be None')

        output_path = output_dir / filename
        save_as_tif(image_data, output_path, compression)
        saved_paths.append(output_path)

    return saved_paths
