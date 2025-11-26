"""CZI metadata extraction utilities.

Provides detailed metadata extraction from Zeiss CZI files using czitools.
Falls back to basic extraction if czitools is not available.
"""

from typing import Any, cast

from czifile import CziFile
from czitools.metadata_tools.boundingbox import CziBoundingBox
from czitools.metadata_tools.channel import CziChannelInfo
from czitools.metadata_tools.detector import CziDetector
from czitools.metadata_tools.dimension import CziDimensions
from czitools.metadata_tools.microscope import CziMicroscope
from czitools.metadata_tools.objective import CziObjectives
from czitools.metadata_tools.scaling import CziScaling
from defusedxml.ElementTree import fromstring


def get_czi_metadata(
    path: str,
    phase_index: int | None = None,
    num_phases: int | None = None,
) -> dict[str, Any]:
    """Extract metadata from CZI file.

    Args:
        path: Path to CZI file
        phase_index: Index of current phase (for multi-phase files)
        num_phases: Total number of phases (for channel splitting)

    Returns:
        Dictionary with Dimensions, Scaling, and other metadata
    """
    try:
        return _extract_with_czitools(path, phase_index, num_phases)
    except ImportError:
        return _extract_basic(path)


def _extract_with_czitools(
    path: str,
    phase_index: int | None = None,
    num_phases: int | None = None,
) -> dict[str, Any]:
    """Extract detailed metadata using czitools library."""

    # Metadata classes to instantiate
    metadata_classes = [
        CziChannelInfo,
        CziDimensions,
        CziScaling,
        CziObjectives,
        CziDetector,
        CziMicroscope,
        CziBoundingBox,
    ]

    combined_metadata: dict[str, dict[str, Any]] = {}

    # Extract metadata from each class
    for metadata_class in metadata_classes:
        class_name = metadata_class.__name__.replace('Czi', '')

        metadata_instance = metadata_class(path)
        metadata_dict = vars(metadata_instance)

        # Replace empty lists/dicts with None
        for key in metadata_dict:
            if (isinstance(metadata_dict[key], list) and len(metadata_dict[key]) == 0) or (
                isinstance(metadata_dict[key], dict) and len(metadata_dict[key]) == 0
            ):
                metadata_dict[key] = None

        # Handle channel info for multi-phase files
        if class_name == 'ChannelInfo' and phase_index is not None and num_phases is not None:
            metadata_dict = _split_channel_info_by_phase(metadata_dict, phase_index, num_phases)

        # Restructure ChannelInfo into per-channel parameters
        if class_name == 'ChannelInfo':
            metadata_dict = _restructure_channel_info(metadata_dict)

        combined_metadata[class_name] = metadata_dict

    # Move 'czisource' to FileInfo and clean up
    file_info = {'czisource': path}
    for value in combined_metadata.values():
        value.pop('czisource', None)
    combined_metadata['FileInfo'] = file_info

    return combined_metadata


def _split_channel_info_by_phase(
    metadata_dict: dict[str, Any],
    phase_index: int,
    num_phases: int,
) -> dict[str, Any]:
    """Split channel info for multi-phase CZI files.

    Args:
        metadata_dict: Channel info metadata
        phase_index: Current phase index
        num_phases: Total number of phases

    Returns:
        Metadata dict with channels for this phase only
    """
    # Extract every nth channel starting from phase_index
    for key in ['names', 'dyes', 'colors', 'clims', 'gamma']:
        if metadata_dict.get(key):
            metadata_dict[key] = metadata_dict[key][phase_index::num_phases]

    return metadata_dict


def _restructure_channel_info(metadata_dict: dict[str, Any]) -> dict[str, Any]:
    """Restructure channel info from lists to per-channel dicts.

    Args:
        metadata_dict: Channel info with lists

    Returns:
        Restructured metadata with Name1, Dye1, Color1, etc.
    """
    required_keys = ['names', 'dyes', 'colors', 'clims', 'gamma']

    # Check if all required keys exist
    if not all(key in metadata_dict for key in required_keys):
        return metadata_dict

    # Verify all lists have the same length
    num_channels = len(metadata_dict['names'])
    if not all(len(metadata_dict[key]) == num_channels for key in required_keys if metadata_dict[key]):
        return metadata_dict

    # Create per-channel parameters
    channel_params = {}
    for i in range(num_channels):
        channel_params[f'Name{i + 1}'] = metadata_dict['names'][i]
        channel_params[f'Dye{i + 1}'] = metadata_dict['dyes'][i]
        channel_params[f'Color{i + 1}'] = metadata_dict['colors'][i]
        channel_params[f'Clims{i + 1}'] = metadata_dict['clims'][i]
        channel_params[f'Gamma{i + 1}'] = metadata_dict['gamma'][i]

    # Update metadata and remove original lists
    metadata_dict.update(channel_params)
    for key in required_keys:
        metadata_dict.pop(key, None)

    return metadata_dict


def _extract_basic(path: str) -> dict[str, dict[str, Any]]:
    """Extract basic metadata without czitools (fallback).

    This provides minimal metadata when czitools is not installed.
    """

    with CziFile(path) as czi:
        axes = cast(list[str], czi.axes)
        shape = cast(list[int], czi.shape)
        dimension_map = dict(zip(axes, shape, strict=True))

        metadata: dict[str, dict[str, Any]] = {
            'Dimensions': {},
            'Scaling': {},
            'FileInfo': {'czisource': path},
        }

        # Get dimensions
        for axis, size in dimension_map.items():
            if axis in 'XYZCT':
                metadata['Dimensions'][f'Size{axis}'] = size

        # Try to extract scaling from XML metadata
        try:
            xml_str = czi.metadata()

            if not xml_str:
                metadata['Scaling'] = {'X': 1.0, 'Y': 1.0, 'Z': 1.0, 'T': 1.0}
                return metadata

            root = fromstring(xml_str)

            ns = {'czi': 'http://www.zeiss.com/microscopy/productdata/schemas/2012/czi'}

            scaling = root.find('.//czi:Scaling', ns)

            for item in scaling.findall('czi:Items/czi:Distance', ns):
                axis_id = item.get('Id')
                value_node = item.find('czi:Value', ns)

                # GUARD CLAUSE 3: Continue loop if necessary data is missing in the item
                if not axis_id or value_node is None or value_node.text is None:
                    continue  # Skip to the next item in the loop

                value = float(value_node.text)

                # Convert to micrometers
                metadata['Scaling'][axis_id] = value * 1e6

        except Exception:  # noqa: BLE001
            metadata['Scaling'] = {'X': 1.0, 'Y': 1.0, 'Z': 1.0, 'T': 1.0}

        return metadata
