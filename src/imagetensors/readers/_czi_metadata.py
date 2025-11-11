"""CZI metadata extraction utilities.

Provides detailed metadata extraction from Zeiss CZI files using czitools.
Falls back to basic extraction if czitools is not available.
"""

from typing import Optional, cast, List, Dict, Any


def get_czi_metadata(
    path: str,
    phase_index: Optional[int] = None,
    num_phases: Optional[int] = None,
) -> Dict[str, Any]:
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
    phase_index: Optional[int] = None,
    num_phases: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract detailed metadata using czitools library."""
    from czitools.metadata_tools.scaling import CziScaling
    from czitools.metadata_tools.dimension import CziDimensions
    from czitools.metadata_tools.boundingbox import CziBoundingBox
    from czitools.metadata_tools.channel import CziChannelInfo
    from czitools.metadata_tools.objective import CziObjectives
    from czitools.metadata_tools.microscope import CziMicroscope
    from czitools.metadata_tools.detector import CziDetector
    
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
    
    combined_metadata: Dict[str, Dict[str, Any]] = {}
    
    # Extract metadata from each class
    for metadata_class in metadata_classes:
        class_name = metadata_class.__name__.replace('Czi', '')
        
        metadata_instance = metadata_class(path)
        metadata_dict = vars(metadata_instance)
        
        # Replace empty lists/dicts with None
        for key in metadata_dict:
            if isinstance(metadata_dict[key], list) and len(metadata_dict[key]) == 0:
                metadata_dict[key] = None
            elif isinstance(metadata_dict[key], dict) and len(metadata_dict[key]) == 0:
                metadata_dict[key] = None
        
        # Handle channel info for multi-phase files
        if class_name == 'ChannelInfo' and phase_index is not None and num_phases is not None:
            metadata_dict = _split_channel_info_by_phase(
                metadata_dict, 
                phase_index, 
                num_phases
            )
        
        # Restructure ChannelInfo into per-channel parameters
        if class_name == 'ChannelInfo':
            metadata_dict = _restructure_channel_info(metadata_dict)
        
        combined_metadata[class_name] = metadata_dict
    
    # Move 'czisource' to FileInfo and clean up
    file_info = {'czisource': path}
    for key in combined_metadata.keys():
        if 'czisource' in combined_metadata[key]:
            del combined_metadata[key]['czisource']
    combined_metadata['FileInfo'] = file_info
    
    return combined_metadata


def _split_channel_info_by_phase(
    metadata_dict: Dict[str, Any],
    phase_index: int,
    num_phases: int,
) -> Dict[str, Any]:
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
        if key in metadata_dict and metadata_dict[key]:
            metadata_dict[key] = metadata_dict[key][phase_index::num_phases]
    
    return metadata_dict


def _restructure_channel_info(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
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
    if not all(
        len(metadata_dict[key]) == num_channels 
        for key in required_keys 
        if metadata_dict[key]
    ):
        return metadata_dict
    
    # Create per-channel parameters
    channel_params = {}
    for i in range(num_channels):
        channel_params[f"Name{i+1}"] = metadata_dict['names'][i]
        channel_params[f"Dye{i+1}"] = metadata_dict['dyes'][i]
        channel_params[f"Color{i+1}"] = metadata_dict['colors'][i]
        channel_params[f"Clims{i+1}"] = metadata_dict['clims'][i]
        channel_params[f"Gamma{i+1}"] = metadata_dict['gamma'][i]
    
    # Update metadata and remove original lists
    metadata_dict.update(channel_params)
    for key in required_keys:
        if key in metadata_dict:
            del metadata_dict[key]
    
    return metadata_dict


def _extract_basic(path: str) -> Dict[str, Dict[str, Any]]:
    """Extract basic metadata without czitools (fallback).
    
    This provides minimal metadata when czitools is not installed.
    """
    from czifile import CziFile
    
    with CziFile(path) as czi:
        axes = cast(List[str], czi.axes)
        shape = cast(List[int], czi.shape)
        dimension_map = dict(zip(axes, shape))
        
        metadata: Dict[str, Dict[str, Any]] = {
            "Dimensions": {},
            "Scaling": {},
            "FileInfo": {"czisource": path},
        }
        
        # Get dimensions
        for axis, size in dimension_map.items():
            if axis in "XYZCT":
                metadata["Dimensions"][f"Size{axis}"] = size
        
        # Try to extract scaling from XML metadata
        try:
            import xml.etree.ElementTree as ET
            xml_str = czi.metadata()

            if not xml_str:
                raise ValueError("Metadata string is empty or None")
            
            root = ET.fromstring(xml_str)
            
            ns = {'czi': 'http://www.zeiss.com/microscopy/productdata/schemas/2012/czi'}
            
            scaling = root.find('.//czi:Scaling', ns)

            if scaling is None:
                # If scaling node is missing, jump straight to the except block's fallback
                raise ValueError("Scaling node not found in metadata")
            
            for item in scaling.findall('czi:Items/czi:Distance', ns):
                axis_id = item.get('Id')
                value_node = item.find('czi:Value', ns)
                
                # GUARD CLAUSE 3: Continue loop if necessary data is missing in the item
                if not axis_id or value_node is None or value_node.text is None:
                    continue # Skip to the next item in the loop

                value = float(value_node.text)
                
                # Convert to micrometers
                metadata["Scaling"][axis_id] = value * 1e6

        except Exception:
            # Fallback values if XML parsing fails
            metadata["Scaling"] = {"X": 1.0, "Y": 1.0, "Z": 1.0, "T": 1.0}
        
        return metadata