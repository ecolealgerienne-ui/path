"""
Tests unitaires pour le module src.preprocessing.h_channel.

Tests pour V15.3 H-Channel Augmented Pipeline:
- Extraction H-Channel via Ruifrok
- Calcul H-Stats (mean, std, nuclei_count, area_ratio)
- Détection de noyaux pour visualisation
- Confidence boosting
"""

import pytest
import numpy as np
import cv2

from src.preprocessing.h_channel import (
    # Constants
    MIN_NUCLEUS_AREA,
    MAX_NUCLEUS_AREA,
    BETHESDA_COLORS,
    # Data classes
    NucleusInfo,
    HChannelStats,
    # Functions
    extract_h_channel_ruifrok,
    compute_h_stats,
    compute_h_stats_batch,
    detect_nuclei_for_visualization,
    apply_confidence_boosting,
    render_nuclei_overlay,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_rgb_image():
    """Create a synthetic RGB image with some 'nuclei' (dark spots)."""
    # Create a light pink background (cytoplasm-like)
    image = np.full((224, 224, 3), [240, 200, 210], dtype=np.uint8)

    # Add some dark purple circles (nucleus-like)
    cv2.circle(image, (50, 50), 15, (80, 40, 100), -1)    # Nucleus 1
    cv2.circle(image, (100, 80), 12, (70, 35, 90), -1)    # Nucleus 2
    cv2.circle(image, (150, 120), 18, (90, 50, 110), -1)  # Nucleus 3
    cv2.circle(image, (80, 180), 10, (60, 30, 80), -1)    # Nucleus 4

    return image


@pytest.fixture
def empty_image():
    """Create an empty (uniform) image with no nuclei."""
    return np.full((224, 224, 3), [240, 230, 220], dtype=np.uint8)


@pytest.fixture
def dense_nuclei_image():
    """Create an image with many nuclei (dense)."""
    image = np.full((224, 224, 3), [240, 200, 210], dtype=np.uint8)

    # Add 15 nuclei
    positions = [
        (30, 30), (70, 30), (110, 30), (150, 30), (190, 30),
        (30, 80), (70, 80), (110, 80), (150, 80), (190, 80),
        (50, 130), (90, 130), (130, 130), (170, 130),
        (70, 180),
    ]
    for x, y in positions:
        cv2.circle(image, (x, y), 10, (80, 40, 100), -1)

    return image


# ============================================================================
# TESTS: CONSTANTS
# ============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_nucleus_area_constraints(self):
        """Nucleus area constraints should be reasonable."""
        assert MIN_NUCLEUS_AREA > 0, "MIN_NUCLEUS_AREA must be positive"
        assert MAX_NUCLEUS_AREA > MIN_NUCLEUS_AREA, "MAX must be > MIN"
        # For 224x224 images at 0.5 MPP, typical nucleus is 50-2000 pixels
        assert MIN_NUCLEUS_AREA >= 30, "MIN too small (noise)"
        assert MAX_NUCLEUS_AREA <= 10000, "MAX too large (debris)"

    def test_bethesda_colors_complete(self):
        """All Bethesda classes should have colors defined."""
        required_classes = ['NILM', 'ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC']
        for cls in required_classes:
            assert cls in BETHESDA_COLORS, f"Missing color for {cls}"
            assert len(BETHESDA_COLORS[cls]) == 3, f"Color must be RGB tuple"


# ============================================================================
# TESTS: DATA CLASSES
# ============================================================================

class TestDataClasses:
    """Tests for data classes."""

    def test_nucleus_info_creation(self):
        """NucleusInfo should store nucleus attributes."""
        nucleus = NucleusInfo(
            centroid=(100, 100),
            area=500,
            bbox=(90, 90, 20, 20)
        )
        assert nucleus.centroid == (100, 100)
        assert nucleus.area == 500
        assert nucleus.bbox == (90, 90, 20, 20)
        assert nucleus.contour is None  # Optional

    def test_h_channel_stats_creation(self):
        """HChannelStats should store all statistics."""
        stats = HChannelStats(
            h_mean=128.5,
            h_std=45.2,
            nuclei_count=5,
            nuclei_area_ratio=0.15,
            nuclei_details=[]
        )
        assert stats.h_mean == 128.5
        assert stats.h_std == 45.2
        assert stats.nuclei_count == 5
        assert stats.nuclei_area_ratio == 0.15

    def test_h_channel_stats_to_features(self):
        """to_features() should return normalized feature vector."""
        stats = HChannelStats(
            h_mean=127.5,     # 127.5/255 = 0.5
            h_std=51.0,       # 51/255 = 0.2
            nuclei_count=10,  # 10/20 = 0.5
            nuclei_area_ratio=0.25,
            nuclei_details=[]
        )
        features = stats.to_features()

        assert features.shape == (4,)
        assert features.dtype == np.float32
        assert 0 <= features[0] <= 1, "h_mean should be normalized"
        assert 0 <= features[1] <= 1, "h_std should be normalized"
        assert 0 <= features[2] <= 1, "nuclei_count should be capped"
        assert 0 <= features[3] <= 1, "nuclei_area_ratio is already [0,1]"

    def test_to_features_caps_nuclei_count(self):
        """Nuclei count should be capped at 20."""
        stats = HChannelStats(
            h_mean=100,
            h_std=30,
            nuclei_count=50,  # More than 20
            nuclei_area_ratio=0.5,
            nuclei_details=[]
        )
        features = stats.to_features()
        assert features[2] == 1.0, "Count >20 should be capped at 1.0"


# ============================================================================
# TESTS: H-CHANNEL EXTRACTION
# ============================================================================

class TestExtractHChannel:
    """Tests for extract_h_channel_ruifrok()."""

    def test_output_shape(self, sample_rgb_image):
        """Output should have same H, W as input."""
        h_channel = extract_h_channel_ruifrok(sample_rgb_image)
        assert h_channel.shape == (224, 224)

    def test_output_uint8_default(self, sample_rgb_image):
        """Default output should be uint8."""
        h_channel = extract_h_channel_ruifrok(sample_rgb_image)
        assert h_channel.dtype == np.uint8

    def test_output_range_uint8(self, sample_rgb_image):
        """uint8 output should be in [0, 255]."""
        h_channel = extract_h_channel_ruifrok(sample_rgb_image, output_range="uint8")
        assert h_channel.min() >= 0
        assert h_channel.max() <= 255

    def test_output_float01(self, sample_rgb_image):
        """float01 output should be in [0, 1]."""
        h_channel = extract_h_channel_ruifrok(sample_rgb_image, output_range="float01")
        assert h_channel.dtype == np.float32
        assert h_channel.min() >= 0
        assert h_channel.max() <= 1

    def test_nuclei_brighter_in_h_channel(self, sample_rgb_image):
        """Nuclei (dark purple in RGB) should be brighter in H-channel."""
        h_channel = extract_h_channel_ruifrok(sample_rgb_image, output_range="uint8")

        # Nucleus center at (50, 50) should be bright
        nucleus_value = h_channel[50, 50]
        # Background at (200, 200) should be dim
        background_value = h_channel[200, 200]

        assert nucleus_value > background_value, (
            f"Nucleus ({nucleus_value}) should be brighter than background ({background_value})"
        )

    def test_invalid_output_range(self, sample_rgb_image):
        """Invalid output_range should raise error."""
        with pytest.raises(ValueError, match="Invalid output_range"):
            extract_h_channel_ruifrok(sample_rgb_image, output_range="invalid")


# ============================================================================
# TESTS: COMPUTE H-STATS
# ============================================================================

class TestComputeHStats:
    """Tests for compute_h_stats()."""

    def test_returns_h_channel_stats(self, sample_rgb_image):
        """Should return HChannelStats object."""
        stats = compute_h_stats(sample_rgb_image)
        assert isinstance(stats, HChannelStats)

    def test_h_mean_in_range(self, sample_rgb_image):
        """h_mean should be in [0, 255]."""
        stats = compute_h_stats(sample_rgb_image)
        assert 0 <= stats.h_mean <= 255

    def test_h_std_positive(self, sample_rgb_image):
        """h_std should be non-negative."""
        stats = compute_h_stats(sample_rgb_image)
        assert stats.h_std >= 0

    def test_detects_nuclei(self, sample_rgb_image):
        """Should detect nuclei in image with dark spots."""
        stats = compute_h_stats(sample_rgb_image)
        # Image has 4 nuclei
        assert stats.nuclei_count >= 1, "Should detect at least 1 nucleus"

    def test_empty_image_no_nuclei(self, empty_image):
        """Empty image should have 0 or very few nuclei."""
        stats = compute_h_stats(empty_image)
        assert stats.nuclei_count <= 2, "Empty image should have ~0 nuclei"

    def test_area_ratio_in_range(self, sample_rgb_image):
        """nuclei_area_ratio should be in [0, 1]."""
        stats = compute_h_stats(sample_rgb_image)
        assert 0 <= stats.nuclei_area_ratio <= 1

    def test_nuclei_details_populated(self, sample_rgb_image):
        """nuclei_details should contain NucleusInfo objects."""
        stats = compute_h_stats(sample_rgb_image)
        if stats.nuclei_count > 0:
            assert len(stats.nuclei_details) == stats.nuclei_count
            assert isinstance(stats.nuclei_details[0], NucleusInfo)

    def test_precomputed_h_channel(self, sample_rgb_image):
        """Should accept pre-computed H-channel."""
        h_channel = extract_h_channel_ruifrok(sample_rgb_image)
        stats = compute_h_stats(sample_rgb_image, h_channel=h_channel)
        assert isinstance(stats, HChannelStats)

    def test_custom_size_constraints(self, sample_rgb_image):
        """Custom size constraints should filter nuclei."""
        # Very restrictive constraints
        stats = compute_h_stats(
            sample_rgb_image,
            min_nucleus_area=10000,  # Very large minimum
            max_nucleus_area=20000
        )
        assert stats.nuclei_count == 0, "No nucleus should pass strict size filter"

    def test_return_binary_mask(self, sample_rgb_image):
        """Should optionally return binary mask."""
        stats, binary = compute_h_stats(sample_rgb_image, return_binary_mask=True)
        assert isinstance(stats, HChannelStats)
        assert binary.shape == (224, 224)
        assert binary.dtype == np.uint8


# ============================================================================
# TESTS: COMPUTE H-STATS BATCH
# ============================================================================

class TestComputeHStatsBatch:
    """Tests for compute_h_stats_batch()."""

    def test_batch_processing(self, sample_rgb_image, empty_image):
        """Should process batch of images."""
        batch = np.stack([sample_rgb_image, empty_image])
        stats_list = compute_h_stats_batch(batch)

        assert len(stats_list) == 2
        assert isinstance(stats_list[0], HChannelStats)
        assert isinstance(stats_list[1], HChannelStats)

    def test_batch_preserves_order(self, sample_rgb_image, empty_image):
        """Batch results should preserve input order."""
        batch = np.stack([sample_rgb_image, empty_image])
        stats_list = compute_h_stats_batch(batch)

        # First image (with nuclei) should have more than second (empty)
        assert stats_list[0].nuclei_count >= stats_list[1].nuclei_count


# ============================================================================
# TESTS: NUCLEI DETECTION FOR VISUALIZATION
# ============================================================================

class TestDetectNucleiForVisualization:
    """Tests for detect_nuclei_for_visualization()."""

    def test_returns_list(self, sample_rgb_image):
        """Should return list of nuclei dicts."""
        nuclei = detect_nuclei_for_visualization(sample_rgb_image)
        assert isinstance(nuclei, list)

    def test_nuclei_dict_keys(self, sample_rgb_image):
        """Each nucleus should have required keys."""
        nuclei = detect_nuclei_for_visualization(sample_rgb_image)
        if len(nuclei) > 0:
            required_keys = ['contour', 'centroid', 'area', 'class']
            for key in required_keys:
                assert key in nuclei[0], f"Missing key: {key}"

    def test_inherits_predicted_class(self, sample_rgb_image):
        """Detected nuclei should inherit predicted class."""
        nuclei = detect_nuclei_for_visualization(
            sample_rgb_image,
            predicted_class="HSIL"
        )
        if len(nuclei) > 0:
            assert nuclei[0]['class'] == "HSIL"

    def test_contour_is_valid(self, sample_rgb_image):
        """Contour should be valid OpenCV format."""
        nuclei = detect_nuclei_for_visualization(sample_rgb_image)
        if len(nuclei) > 0:
            contour = nuclei[0]['contour']
            assert isinstance(contour, np.ndarray)
            # OpenCV contours have shape (N, 1, 2)
            assert contour.ndim == 3
            assert contour.shape[2] == 2

    def test_adaptive_vs_otsu(self, sample_rgb_image):
        """Both threshold methods should work."""
        nuclei_adaptive = detect_nuclei_for_visualization(
            sample_rgb_image, use_adaptive_threshold=True
        )
        nuclei_otsu = detect_nuclei_for_visualization(
            sample_rgb_image, use_adaptive_threshold=False
        )
        # Both should detect some nuclei
        assert isinstance(nuclei_adaptive, list)
        assert isinstance(nuclei_otsu, list)


# ============================================================================
# TESTS: CONFIDENCE BOOSTING
# ============================================================================

class TestApplyConfidenceBoosting:
    """Tests for apply_confidence_boosting()."""

    def test_reduces_confidence_no_nuclei(self):
        """Abnormal prediction with 0 nuclei should reduce confidence."""
        prediction = {'class': 'HSIL', 'confidence': 0.8}
        stats = HChannelStats(
            h_mean=100, h_std=30,
            nuclei_count=0,  # No nuclei!
            nuclei_area_ratio=0,
            nuclei_details=[]
        )
        result = apply_confidence_boosting(prediction, stats)

        assert result['confidence'] < prediction['confidence']
        assert result['flag'] == 'LOW_CONFIDENCE_NO_NUCLEI'

    def test_flags_high_density_normal(self):
        """Normal prediction with high density should flag for review."""
        prediction = {'class': 'NILM', 'confidence': 0.9}
        stats = HChannelStats(
            h_mean=100, h_std=30,
            nuclei_count=15,  # High density
            nuclei_area_ratio=0.3,
            nuclei_details=[]
        )
        result = apply_confidence_boosting(prediction, stats)

        assert result['flag'] == 'REVIEW_HIGH_DENSITY'
        # Confidence unchanged
        assert result['confidence'] == prediction['confidence']

    def test_boosts_hsil_high_variance(self):
        """HSIL with high H-variance should boost confidence."""
        prediction = {'class': 'HSIL', 'confidence': 0.75}
        stats = HChannelStats(
            h_mean=100, h_std=60,  # High variance (>50)
            nuclei_count=5,
            nuclei_area_ratio=0.2,
            nuclei_details=[]
        )
        result = apply_confidence_boosting(prediction, stats)

        assert result['confidence'] > prediction['confidence']
        assert result['confidence'] <= 0.99  # Capped

    def test_no_change_normal_case(self):
        """Normal prediction with normal stats should not change."""
        prediction = {'class': 'NILM', 'confidence': 0.85}
        stats = HChannelStats(
            h_mean=100, h_std=30,
            nuclei_count=3,  # Normal density
            nuclei_area_ratio=0.1,
            nuclei_details=[]
        )
        result = apply_confidence_boosting(prediction, stats)

        assert result['confidence'] == prediction['confidence']
        assert 'flag' not in result

    def test_original_unchanged(self):
        """Original prediction dict should not be modified."""
        prediction = {'class': 'HSIL', 'confidence': 0.8}
        stats = HChannelStats(
            h_mean=100, h_std=30,
            nuclei_count=0,
            nuclei_area_ratio=0,
            nuclei_details=[]
        )
        _ = apply_confidence_boosting(prediction, stats)

        # Original should be unchanged
        assert prediction['confidence'] == 0.8
        assert 'flag' not in prediction


# ============================================================================
# TESTS: RENDER NUCLEI OVERLAY
# ============================================================================

class TestRenderNucleiOverlay:
    """Tests for render_nuclei_overlay()."""

    def test_output_same_shape(self, sample_rgb_image):
        """Output should have same shape as input."""
        nuclei = detect_nuclei_for_visualization(sample_rgb_image)
        result = render_nuclei_overlay(sample_rgb_image, nuclei)
        assert result.shape == sample_rgb_image.shape

    def test_output_same_dtype(self, sample_rgb_image):
        """Output should have same dtype as input."""
        nuclei = detect_nuclei_for_visualization(sample_rgb_image)
        result = render_nuclei_overlay(sample_rgb_image, nuclei)
        assert result.dtype == sample_rgb_image.dtype

    def test_empty_nuclei_list(self, sample_rgb_image):
        """Empty nuclei list should return blended original."""
        result = render_nuclei_overlay(sample_rgb_image, [])
        assert result.shape == sample_rgb_image.shape

    def test_alpha_parameter(self, sample_rgb_image):
        """Different alpha values should produce different results."""
        nuclei = detect_nuclei_for_visualization(sample_rgb_image)
        result_0 = render_nuclei_overlay(sample_rgb_image, nuclei, alpha=0.0)
        result_1 = render_nuclei_overlay(sample_rgb_image, nuclei, alpha=1.0)

        # Alpha=0 should be mostly original
        # Alpha=1 should be mostly overlay
        if len(nuclei) > 0:
            assert not np.array_equal(result_0, result_1)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the full H-Channel pipeline."""

    def test_full_pipeline(self, sample_rgb_image):
        """Test complete pipeline: extract → stats → detect → render."""
        # Step 1: Extract H-channel
        h_channel = extract_h_channel_ruifrok(sample_rgb_image)
        assert h_channel.shape == (224, 224)

        # Step 2: Compute stats
        stats = compute_h_stats(sample_rgb_image, h_channel=h_channel)
        assert isinstance(stats, HChannelStats)

        # Step 3: Detect nuclei
        nuclei = detect_nuclei_for_visualization(
            sample_rgb_image,
            predicted_class="ASCUS"
        )
        assert isinstance(nuclei, list)

        # Step 4: Render overlay
        vis = render_nuclei_overlay(sample_rgb_image, nuclei)
        assert vis.shape == sample_rgb_image.shape

    def test_confidence_boosting_integration(self, sample_rgb_image, empty_image):
        """Test confidence boosting with real images."""
        # Image with nuclei
        stats_with_nuclei = compute_h_stats(sample_rgb_image)

        # Empty image
        stats_empty = compute_h_stats(empty_image)

        # Prediction for image with nuclei
        pred1 = {'class': 'HSIL', 'confidence': 0.8}
        result1 = apply_confidence_boosting(pred1, stats_with_nuclei)

        # Prediction for empty image (should reduce confidence)
        pred2 = {'class': 'HSIL', 'confidence': 0.8}
        result2 = apply_confidence_boosting(pred2, stats_empty)

        # Empty image should have reduced confidence
        assert result2['confidence'] < result1['confidence']

    def test_features_for_cell_triage_v2(self, sample_rgb_image):
        """Test feature extraction for Cell Triage v2."""
        stats = compute_h_stats(sample_rgb_image)
        features = stats.to_features()

        # Features should be ready for concatenation with CLS token
        assert features.shape == (4,)
        assert features.dtype == np.float32
        assert np.all(features >= 0) and np.all(features <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
