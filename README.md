<p align="center">
  <img src="https://github.com/bryanbarcelona/imagetensors/raw/main/images/logo.png" alt="Image Tensors" width="30%">
</p>

## 🔬 imagetensors: Unified Microscopy Image Reading

**`imagetensors`** provides a standardized, unified interface for reading multiple proprietary microscopy image formats, converting them into **standardized 5D NumPy arrays (T, Z, C, Y, X)** for seamless downstream analysis and processing.

It aims to abstract away the complexity of handling diverse file formats, allowing researchers and developers to focus on image data analysis.

-----

### ✨ Features

  * **Unified Interface:** A single `ImageReader` class handles all supported formats automatically based on the file extension.
  * **Standardized Output:** All images are read into a consistent **5D array format (T-ime, Z-slice, C-hannel, Y-row, X-column)**.
  * **Comprehensive Metadata:** Extracts essential physical and dimensional metadata, including pixel/voxel size, time resolution, and display ranges.
  * **ImageJ Compatibility:** Built-in utilities to save images as **ImageJ-compatible TIFF** files with embedded metadata.
  * **Extensible Architecture:** Based on an Abstract Base Class (`BaseImageReader`) for easy addition of new formats.

-----

### 💾 Supported Formats

| Extension | Format Description | Reader Class |
| :--- | :--- | :--- |
| `.lif` | Leica Image File (supports multi-series) | `LifImageReader` |
| `.czi` | Zeiss CZI (supports phase separation) | `CziImageReader` |
| `.oib` | Olympus OIB | `OibImageReader` |
| `.nd` | Nikon ND (supports multi-file assemblies) | `NdImageReader` |
| `.tif`, `.tiff` | Standard TIFF Stacks | `TifImageReader` |

-----

### ⬇️ Installation

You can install `imagetensors` using various Python package managers or by cloning the repository.

#### Using Pip or uv

The most common way to install is via PyPI using `pip` or the faster alternative, `uv`:

```bash
# Using pip
pip install imagetensors

# Using uv
uv add imagetensors
```

#### From Source

If you need the latest development version, you can clone the repository and install it locally:

```bash
git clone https://github.com/bryanbarcelona/imagetensors.git
cd imagetensors
pip install .
```

-----

#### Reading Images

The core public API is the **`ImageReader`** factory class. It automatically selects the correct underlying reader based on the file extension.

```python
from imagetensors import ImageReader
from pathlib import Path

# Initialize the reader for any supported file
file_path = "path/to/my_multi_series_image.lif"
reader = ImageReader(file_path)

# ImageReader is iterable, allowing you to process one series at a time
print(f"Reading file: {reader.path.name}")

for idx, image_data in enumerate(reader):
    # image_data is an ImageData object

    # Access the 5D NumPy array (T, Z, C, Y, X)
    array = image_data.array

    # Access the standardized metadata
    metadata = image_data.metadata

    # Print basic info
    print(f"\nSeries {idx + 1}:")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")
    print(f"  Z-Spacing: {metadata.spacing:.4f} µm")

    # Example usage: process the image data
    # processed_array = my_analysis_function(array)
```

#### The `ImageData` Object

All readers yield an **`ImageData`** dataclass containing the processed 5D array and its corresponding **`Metadata`** object.

```python
@dataclass
class ImageData:
    array: np.ndarray  # Shape: (T, Z, C, Y, X)
    metadata: Metadata
```

-----

### 🖼️ Image Conversion Utilities

The package provides utilities to save the standardized `ImageData` objects as compliant **ImageJ-compatible TIFF** files, embedding critical metadata.

#### Saving a Single Image Series

Use `save_as_tif` to save a single `ImageData` object.

```python
from imagetensors import ImageReader, save_as_tif
from pathlib import Path

file_path = "path/to/my_image.czi"
reader = ImageReader(file_path)

# 1. Get the first image series
first_series = next(iter(reader))

output_path = Path("output_data") / f"{Path(file_path).stem}_series_1.tif"
saved_path = save_as_tif(first_series, output_path, compression='lzw')

print(f"Single image series saved to: {saved_path}")
```

#### Saving All Series in a Batch

Use `save_all_as_tif` to process a multi-series file and save every series into a specified output directory.

```python
from imagetensors import ImageReader, save_all_as_tif
from pathlib import Path

multi_series_file = "path/to/multi_series_image.lif"
reader = ImageReader(multi_series_file)

output_directory = "batch_tiff_output"

# Save all series using a template name
saved_paths = save_all_as_tif(
    reader=reader,
    output_dir=output_directory,
    # Template uses {stem} of the original file and {series} index
    name_template="{stem}_S{series}.tif",
    compression='zlib'
)

print(f"Batch conversion complete. {len(saved_paths)} files saved in: {output_directory}")
# Example output filename: multi_series_image_S1.tif, multi_series_image_S2.tif, etc.
```

-----

### ⚙️ Advanced Configuration

#### Overriding Physical Pixel Size

You can optionally override the pixel size read from the file's metadata during reader initialization. This can be useful for files with missing or incorrect metadata.

```python
# Force all images from this file to use a 0.1 µm pixel size
reader = ImageReader(
    file_path="path/to/image.oib",
    override_pixel_size_um=0.1
)

for image_data in reader:
    # image_data.metadata.x_resolution and y_resolution will be 0.1
    ...
```

-----

### 🤖 AI Usage Declaration

Look, I’m not proud of it, but yeah...an AI helped scribble parts of this README. I’m basically a human who can’t type fast enough. I went through every word afterward, fixed the weird bits, and made sure it still sounded like the mess I call “me.” It’s less “collaboration” and more “I needed a ghostwriter who doesn’t charge by the hour.”
