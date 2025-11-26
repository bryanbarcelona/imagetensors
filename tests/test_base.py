import numpy as np
import pytest  # type: ignore

from imagetensors.base import BaseImageReader
from imagetensors.models import ImageData, Metadata


# Test concrete implementation for testing abstract class
class ConcreteImageReader(BaseImageReader):
    """Concrete implementation for testing BaseImageReader."""

    def read(self):
        # Mock implementation that yields dummy ImageData
        mock_array = np.random.rand(1, 1, 1, 10, 10).astype(np.uint16)
        yield ImageData(array=mock_array, metadata=Metadata(image_name='value'))


class TestBaseImageReader:
    """Test BaseImageReader abstract class."""

    def test_initialization_with_valid_path(self, tmp_path):
        """Test initialization with valid file path."""
        # Create a temporary file
        test_file = tmp_path / 'test_image.tif'
        test_file.touch()

        reader = ConcreteImageReader(str(test_file))
        assert reader.path == test_file.resolve()
        assert reader._override_pixel_size_um is None

    def test_initialization_with_override_pixel_size(self, tmp_path):
        """Test initialization with pixel size override."""
        test_file = tmp_path / 'test_image.tif'
        test_file.touch()

        reader = ConcreteImageReader(str(test_file), override_pixel_size_um=0.25)
        assert reader._override_pixel_size_um == 0.25

    def test_initialization_with_nonexistent_file(self):
        """Test initialization raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ConcreteImageReader('nonexistent_file.tif')

        assert 'Image file not found' in str(exc_info.value)

    def test_path_resolution(self, tmp_path):
        """Test that path is resolved to absolute path."""
        test_file = tmp_path / 'test_image.tif'
        test_file.touch()

        reader = ConcreteImageReader(str(test_file))
        assert reader.path.is_absolute()
        assert reader.path.name == 'test_image.tif'

    def test_iter_method(self, tmp_path):
        """Test that __iter__ returns iterator from read method."""
        test_file = tmp_path / 'test_image.tif'
        test_file.touch()

        reader = ConcreteImageReader(str(test_file))
        iterator = iter(reader)

        # Should be an iterator (generator)
        assert hasattr(iterator, '__next__')

        # Should yield ImageData objects
        image_data = next(iterator)
        assert isinstance(image_data, ImageData)


class TestBuildInfoString:
    """Test _build_info_string method."""

    @pytest.fixture
    def sample_array(self):
        """Create sample 5D array for testing."""
        return np.random.rand(2, 3, 4, 50, 100).astype(np.float32)

    @pytest.fixture
    def sample_reader(self, tmp_path):
        """Create a reader instance for testing."""
        test_file = tmp_path / 'test.tif'
        test_file.touch()
        return ConcreteImageReader(str(test_file))

    def test_build_info_string_basic(self, sample_reader, sample_array):
        """Test basic info string generation."""
        info_string = sample_reader._build_info_string(sample_array)

        # Check basic dimensions
        assert 'SizeT = 2' in info_string
        assert 'SizeZ = 3' in info_string
        assert 'SizeC = 4' in info_string
        assert 'SizeY = 50' in info_string
        assert 'SizeX = 100' in info_string
        assert 'DimensionOrder = TZCYX' in info_string

    def test_build_info_string_dtype_and_bits(self, sample_reader):
        """Test info string with different data types."""
        # Test uint8
        uint8_array = np.random.rand(1, 1, 1, 10, 10).astype(np.uint8)
        info_string = sample_reader._build_info_string(uint8_array)
        assert 'BitsPerPixel = 8' in info_string
        assert 'PixelType = uint8' in info_string

        # Test uint16
        uint16_array = np.random.rand(1, 1, 1, 10, 10).astype(np.uint16)
        info_string = sample_reader._build_info_string(uint16_array)
        assert 'BitsPerPixel = 16' in info_string
        assert 'PixelType = uint16' in info_string

        # Test float32
        float32_array = np.random.rand(1, 1, 1, 10, 10).astype(np.float32)
        info_string = sample_reader._build_info_string(float32_array)
        assert 'BitsPerPixel = 32' in info_string
        assert 'PixelType = float32' in info_string

    def test_build_info_string_endianness(self, sample_reader):
        """Test endianness detection in info string."""
        # Test little endian
        little_endian_array = np.array([1, 2, 3], dtype='<i4')
        info_string = sample_reader._build_info_string(little_endian_array.reshape(1, 1, 1, 1, 3))
        assert 'LittleEndian = true' in info_string

        # Test big endian
        big_endian_array = np.array([1, 2, 3], dtype='>i4')
        info_string = sample_reader._build_info_string(big_endian_array.reshape(1, 1, 1, 1, 3))
        assert 'LittleEndian = false' in info_string

    def test_build_info_string_with_config(self, sample_reader, sample_array):
        """Test info string generation with configuration."""
        config = {
            'channels': {'names': ['DAPI', 'GFP', 'RFP'], 'exposure': '100ms'},
            'acquisition': {'date': '2024-01-01'},
        }

        info_string = sample_reader._build_info_string(sample_array, config)

        # Check flattened config entries
        assert "[channels][names] = ['DAPI', 'GFP', 'RFP']" in info_string
        assert '[channels][exposure] = 100ms' in info_string
        assert '[acquisition][date] = 2024-01-01' in info_string

    def test_build_info_string_with_none_config(self, sample_reader, sample_array):
        """Test info string generation with None config."""
        info_string = sample_reader._build_info_string(sample_array, None)

        # Should still contain basic info
        assert 'DimensionOrder = TZCYX' in info_string
        assert 'IsInterleaved = false' in info_string


class TestFlattenConfig:
    """Test _flatten_config method."""

    @pytest.fixture
    def sample_reader(self, tmp_path):
        test_file = tmp_path / 'test.tif'
        test_file.touch()
        return ConcreteImageReader(str(test_file))

    def test_flatten_config_simple(self, sample_reader):
        """Test flattening simple configuration."""
        config = {'key1': 'value1', 'key2': 'value2'}

        result = sample_reader._flatten_config(config)

        assert '[key1] = value1' in result
        assert '[key2] = value2' in result
        assert result.endswith('\r\n')  # Should use Windows line endings

    def test_flatten_config_nested(self, sample_reader):
        """Test flattening nested configuration."""
        config = {'level1': {'level2': {'level3': 'deep_value'}, 'simple': 'value'}, 'top': 'top_value'}

        result = sample_reader._flatten_config(config)

        assert '[level1][level2][level3] = deep_value' in result
        assert '[level1][simple] = value' in result
        assert '[top] = top_value' in result

    def test_flatten_config_with_none_values(self, sample_reader):
        """Test flattening configuration with None values."""
        config = {'valid': 'value', 'none_value': None, 'nested': {'also_none': None, 'valid': 'nested_value'}}

        result = sample_reader._flatten_config(config)

        # Should include valid values but skip None values
        assert '[valid] = value' in result
        assert '[nested][valid] = nested_value' in result
        assert 'none_value' not in result
        assert 'also_none' not in result

    def test_flatten_config_sorted_keys(self, sample_reader):
        """Test that keys are sorted alphabetically."""
        config = {'zebra': 'value', 'alpha': 'value', 'beta': 'value'}

        result = sample_reader._flatten_config(config)

        # Check order is alphabetical
        alpha_pos = result.find('[alpha]')
        beta_pos = result.find('[beta]')
        zebra_pos = result.find('[zebra]')

        assert alpha_pos < beta_pos < zebra_pos

    def test_flatten_config_empty(self, sample_reader):
        """Test flattening empty configuration."""
        result = sample_reader._flatten_config({})
        assert not result


def test_abstract_method_implementation():
    """Test that BaseImageReader cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        BaseImageReader('dummy_path')

    assert 'abstract' in str(exc_info.value).lower()


# Parametrized tests for different array shapes
@pytest.mark.parametrize(
    'shape',
    [
        (1, 1, 1, 10, 10),  # Single image
        (5, 1, 1, 10, 10),  # Multiple time points
        (1, 10, 1, 10, 10),  # Multiple Z slices
        (1, 1, 3, 10, 10),  # Multiple channels
        (2, 5, 3, 100, 100),  # Complex case
    ],
)
def test_build_info_string_various_shapes(shape, tmp_path):
    """Test info string generation with various array shapes."""
    test_file = tmp_path / 'test.tif'
    test_file.touch()
    reader = ConcreteImageReader(str(test_file))

    array = np.random.rand(*shape).astype(np.uint16)
    info_string = reader._build_info_string(array)

    # Verify all dimensions are correctly reported
    for i, dim in enumerate(['T', 'Z', 'C', 'Y', 'X']):
        assert f'Size{dim} = {shape[i]}' in info_string
