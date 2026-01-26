"""
Input Router Adaptatif pour traitement multi-format.

Ce module détecte automatiquement le type d'input et applique
le preprocessing approprié pour produire des tiles 224×224
compatibles avec le moteur V13.

Types supportés:
    - PATCH: 256×256 (PanNuke) → Center crop 224×224
    - TILE: 224×224 → Direct (aucune transformation)
    - IMAGE: <10000px → Tiling + filtrage basique
    - WSI: >10000px → CLAM + HistoQC + Tiling complet

Spec: docs/specs/WSI_INDUSTRIAL_PIPELINE_SPEC.md (Section 2.1b)
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Generator, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import cv2


# =============================================================================
# CONSTANTS
# =============================================================================

# Target size for V13 inference
TARGET_SIZE = 224

# PanNuke original size
PANNUKE_SIZE = 256

# Crop offset for center crop (256 - 224) / 2 = 16
CENTER_CROP_OFFSET = (PANNUKE_SIZE - TARGET_SIZE) // 2

# WSI extensions (require OpenSlide)
WSI_EXTENSIONS = {'.svs', '.ndpi', '.mrxs', '.scn', '.vms', '.vmu', '.bif', '.tiff'}

# Array extensions (PanNuke style)
ARRAY_EXTENSIONS = {'.npy', '.npz'}

# Image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

# Threshold for WSI detection (pixels)
WSI_SIZE_THRESHOLD = 10000

# Tissue detection thresholds
TISSUE_RATIO_THRESHOLD = 0.5
BLUR_VARIANCE_THRESHOLD = 100.0
ENTROPY_THRESHOLD = 4.0


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class InputType(Enum):
    """Type d'input détecté."""
    PATCH = "patch"      # 256×256 (PanNuke, CoNSeP)
    TILE = "tile"        # 224×224 (pré-extrait)
    IMAGE = "image"      # <10000px (photo biopsie)
    WSI = "wsi"          # >10000px (lame scanner)
    UNKNOWN = "unknown"


@dataclass
class InputMetadata:
    """Métadonnées d'un input."""
    path: Path
    input_type: InputType
    dimensions: Tuple[int, int]  # (height, width)
    channels: int = 3
    mpp: Optional[float] = None  # Microns per pixel
    format: Optional[str] = None
    file_size_bytes: Optional[int] = None

    @property
    def is_wsi(self) -> bool:
        return self.input_type == InputType.WSI

    @property
    def needs_tiling(self) -> bool:
        return self.input_type in (InputType.IMAGE, InputType.WSI)

    @property
    def estimated_tiles(self) -> int:
        """Estime le nombre de tiles à extraire."""
        if self.input_type == InputType.PATCH:
            return 1
        elif self.input_type == InputType.TILE:
            return 1
        elif self.input_type == InputType.IMAGE:
            h, w = self.dimensions
            return ((h // TARGET_SIZE) + 1) * ((w // TARGET_SIZE) + 1)
        elif self.input_type == InputType.WSI:
            # Estimation grossière: ~30% de la surface est du tissu
            h, w = self.dimensions
            total = ((h // TARGET_SIZE) + 1) * ((w // TARGET_SIZE) + 1)
            return int(total * 0.3)
        return 0


@dataclass
class ProcessedTile:
    """Tile 224×224 prêt pour inference V13."""
    image: np.ndarray  # (224, 224, 3) uint8
    x: int  # Coordonnée x dans l'image source
    y: int  # Coordonnée y dans l'image source
    source_path: Path
    source_type: InputType
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.image.shape == (TARGET_SIZE, TARGET_SIZE, 3), \
            f"Expected (224, 224, 3), got {self.image.shape}"
        assert self.image.dtype == np.uint8, \
            f"Expected uint8, got {self.image.dtype}"


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def detect_input_type(path: Union[str, Path]) -> InputType:
    """
    Détecte automatiquement le type d'input.

    Args:
        path: Chemin vers le fichier

    Returns:
        InputType détecté

    Example:
        >>> detect_input_type("slide.svs")
        InputType.WSI
        >>> detect_input_type("pannuke_001.png")  # 256×256
        InputType.PATCH
    """
    path = Path(path)
    ext = path.suffix.lower()

    # WSI formats (require OpenSlide)
    if ext in WSI_EXTENSIONS:
        # .tiff can be WSI or regular image, check size
        if ext == '.tiff':
            return _detect_tiff_type(path)
        return InputType.WSI

    # Array formats (PanNuke style)
    if ext in ARRAY_EXTENSIONS:
        return InputType.PATCH

    # Image formats - need to check dimensions
    if ext in IMAGE_EXTENSIONS:
        return _detect_image_type(path)

    return InputType.UNKNOWN


def _detect_tiff_type(path: Path) -> InputType:
    """Détecte si un TIFF est un WSI ou une image normale."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            # Check if pyramidal (WSI indicator)
            if len(tif.pages) > 1 or tif.pages[0].is_tiled:
                return InputType.WSI
            # Check dimensions
            shape = tif.pages[0].shape
            if len(shape) >= 2 and max(shape[:2]) > WSI_SIZE_THRESHOLD:
                return InputType.WSI
    except ImportError:
        pass
    except Exception:
        pass
    return InputType.IMAGE


def _detect_image_type(path: Path) -> InputType:
    """Détecte le type d'une image standard (PNG, JPG, etc.)."""
    try:
        # Use cv2 to read just the header
        img = cv2.imread(str(path))
        if img is None:
            return InputType.UNKNOWN

        h, w = img.shape[:2]

        # Exact match for common sizes
        if h == PANNUKE_SIZE and w == PANNUKE_SIZE:
            return InputType.PATCH
        elif h == TARGET_SIZE and w == TARGET_SIZE:
            return InputType.TILE
        elif max(h, w) > WSI_SIZE_THRESHOLD:
            return InputType.WSI
        else:
            return InputType.IMAGE

    except Exception:
        return InputType.UNKNOWN


def get_input_metadata(path: Union[str, Path]) -> InputMetadata:
    """
    Extrait les métadonnées complètes d'un input.

    Args:
        path: Chemin vers le fichier

    Returns:
        InputMetadata avec toutes les informations
    """
    path = Path(path)
    input_type = detect_input_type(path)

    # Get file size
    file_size = path.stat().st_size if path.exists() else None

    # Get dimensions based on type
    if input_type == InputType.WSI:
        dimensions, mpp, fmt = _get_wsi_metadata(path)
    elif input_type in (InputType.PATCH, InputType.TILE, InputType.IMAGE):
        dimensions, mpp, fmt = _get_image_metadata(path)
    else:
        dimensions = (0, 0)
        mpp = None
        fmt = None

    return InputMetadata(
        path=path,
        input_type=input_type,
        dimensions=dimensions,
        mpp=mpp,
        format=fmt,
        file_size_bytes=file_size
    )


def _get_wsi_metadata(path: Path) -> Tuple[Tuple[int, int], Optional[float], str]:
    """Extrait les métadonnées d'un WSI via OpenSlide."""
    try:
        import openslide
        slide = openslide.OpenSlide(str(path))
        dims = slide.dimensions  # (width, height)
        mpp = slide.properties.get('openslide.mpp-x')
        mpp = float(mpp) if mpp else None
        fmt = slide.properties.get('openslide.vendor', 'unknown')
        slide.close()
        return (dims[1], dims[0]), mpp, fmt  # Return (height, width)
    except ImportError:
        return (0, 0), None, "openslide_not_installed"
    except Exception as e:
        return (0, 0), None, f"error: {e}"


def _get_image_metadata(path: Path) -> Tuple[Tuple[int, int], Optional[float], str]:
    """Extrait les métadonnées d'une image standard."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            return (0, 0), None, "unreadable"
        h, w = img.shape[:2]
        fmt = path.suffix.lower().replace('.', '')
        return (h, w), None, fmt
    except Exception as e:
        return (0, 0), None, f"error: {e}"


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def transform_pannuke_to_224(image: np.ndarray, method: str = "center_crop") -> np.ndarray:
    """
    Transforme une image PanNuke 256×256 en 224×224.

    Args:
        image: Image RGB (256, 256, 3) uint8
        method: "center_crop" (recommandé) ou "resize"

    Returns:
        Image RGB (224, 224, 3) uint8

    Spec: WSI_INDUSTRIAL_PIPELINE_SPEC.md Section 2.1b

    Center Crop (recommandé):
        - Préserve les proportions des noyaux
        - Perd 16px de bordure (acceptable)
        - offset = (256-224)/2 = 16
        - crop: image[16:240, 16:240]

    Resize (alternative):
        - Garde tout le contenu
        - Légère déformation (ratio 0.875)
    """
    assert image.shape[:2] == (PANNUKE_SIZE, PANNUKE_SIZE), \
        f"Expected 256×256, got {image.shape[:2]}"

    if method == "center_crop":
        # Center crop: image[16:240, 16:240]
        offset = CENTER_CROP_OFFSET
        cropped = image[offset:offset+TARGET_SIZE, offset:offset+TARGET_SIZE]
        return cropped.copy()

    elif method == "resize":
        # Resize with bilinear interpolation
        resized = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE),
                            interpolation=cv2.INTER_LINEAR)
        return resized

    else:
        raise ValueError(f"Unknown method: {method}. Use 'center_crop' or 'resize'")


def transform_masks_pannuke_to_224(
    masks: np.ndarray,
    method: str = "center_crop"
) -> np.ndarray:
    """
    Transforme les masques PanNuke 256×256 en 224×224.

    Args:
        masks: Masques (256, 256) ou (256, 256, C)
        method: "center_crop" ou "resize"

    Returns:
        Masques (224, 224) ou (224, 224, C)

    Note: Pour les masques, on utilise INTER_NEAREST pour préserver
          les valeurs discrètes (labels d'instances).
    """
    if masks.ndim == 2:
        h, w = masks.shape
    else:
        h, w = masks.shape[:2]

    assert h == PANNUKE_SIZE and w == PANNUKE_SIZE, \
        f"Expected 256×256, got {h}×{w}"

    if method == "center_crop":
        offset = CENTER_CROP_OFFSET
        if masks.ndim == 2:
            return masks[offset:offset+TARGET_SIZE, offset:offset+TARGET_SIZE].copy()
        else:
            return masks[offset:offset+TARGET_SIZE, offset:offset+TARGET_SIZE, :].copy()

    elif method == "resize":
        # Use NEAREST for masks to preserve label values
        if masks.ndim == 2:
            return cv2.resize(masks, (TARGET_SIZE, TARGET_SIZE),
                            interpolation=cv2.INTER_NEAREST)
        else:
            # Resize each channel separately
            resized_channels = []
            for c in range(masks.shape[2]):
                resized = cv2.resize(masks[:, :, c], (TARGET_SIZE, TARGET_SIZE),
                                    interpolation=cv2.INTER_NEAREST)
                resized_channels.append(resized)
            return np.stack(resized_channels, axis=-1)

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

def is_tissue_tile(tile: np.ndarray,
                   tissue_threshold: float = TISSUE_RATIO_THRESHOLD) -> bool:
    """
    Vérifie si un tile contient suffisamment de tissu.

    Args:
        tile: Image RGB (H, W, 3)
        tissue_threshold: Ratio minimum de tissu (default 0.5)

    Returns:
        True si le tile contient assez de tissu
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    # Pixels non-blancs (< 220) sont considérés comme tissu
    tissue_mask = gray < 220
    tissue_ratio = np.mean(tissue_mask)
    return tissue_ratio >= tissue_threshold


def is_focused_tile(tile: np.ndarray,
                    blur_threshold: float = BLUR_VARIANCE_THRESHOLD) -> bool:
    """
    Vérifie si un tile est bien focalisé (pas flou).

    Args:
        tile: Image RGB (H, W, 3)
        blur_threshold: Variance Laplacien minimum (default 100)

    Returns:
        True si le tile est net
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var >= blur_threshold


def compute_entropy(tile: np.ndarray) -> float:
    """
    Calcule l'entropie de Shannon d'un tile.

    Args:
        tile: Image RGB (H, W, 3)

    Returns:
        Entropie (>4.0 = informatif, <4.0 = homogène)
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    # Histogram
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize
    # Shannon entropy
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def filter_tile(tile: np.ndarray,
                check_tissue: bool = True,
                check_focus: bool = True,
                check_entropy: bool = True) -> Tuple[bool, str]:
    """
    Filtre un tile selon plusieurs critères.

    Args:
        tile: Image RGB (H, W, 3)
        check_tissue: Vérifier le ratio de tissu
        check_focus: Vérifier la netteté
        check_entropy: Vérifier l'entropie

    Returns:
        (passed, reason): True si passé, sinon raison du rejet
    """
    if check_tissue and not is_tissue_tile(tile):
        return False, "background"

    if check_focus and not is_focused_tile(tile):
        return False, "blur"

    if check_entropy and compute_entropy(tile) < ENTROPY_THRESHOLD:
        return False, "low_entropy"

    return True, "passed"


# =============================================================================
# INPUT ROUTER CLASS
# =============================================================================

class InputRouter:
    """
    Router adaptatif pour traitement multi-format.

    Détecte automatiquement le type d'input et produit des tiles 224×224
    prêts pour l'inference V13.

    Example:
        >>> router = InputRouter()
        >>>
        >>> # Process PanNuke patch
        >>> for tile in router.process("pannuke_001.png"):
        ...     features = model.forward_features(preprocess_image(tile.image))
        >>>
        >>> # Process WSI (requires OpenSlide)
        >>> for tile in router.process("slide.svs", max_tiles=1000):
        ...     features = model.forward_features(preprocess_image(tile.image))
    """

    def __init__(self,
                 pannuke_method: str = "center_crop",
                 filter_tiles: bool = True,
                 tissue_threshold: float = TISSUE_RATIO_THRESHOLD,
                 blur_threshold: float = BLUR_VARIANCE_THRESHOLD,
                 entropy_threshold: float = ENTROPY_THRESHOLD):
        """
        Args:
            pannuke_method: "center_crop" ou "resize" pour PanNuke 256→224
            filter_tiles: Activer le filtrage des tiles (IMAGE/WSI)
            tissue_threshold: Ratio minimum de tissu
            blur_threshold: Variance Laplacien minimum
            entropy_threshold: Entropie Shannon minimum
        """
        self.pannuke_method = pannuke_method
        self.filter_tiles = filter_tiles
        self.tissue_threshold = tissue_threshold
        self.blur_threshold = blur_threshold
        self.entropy_threshold = entropy_threshold

    def detect_type(self, path: Union[str, Path]) -> InputType:
        """Détecte le type d'input."""
        return detect_input_type(path)

    def get_metadata(self, path: Union[str, Path]) -> InputMetadata:
        """Récupère les métadonnées d'un input."""
        return get_input_metadata(path)

    def process(self,
                path: Union[str, Path],
                max_tiles: Optional[int] = None) -> Generator[ProcessedTile, None, None]:
        """
        Traite un input et génère des tiles 224×224.

        Args:
            path: Chemin vers l'input
            max_tiles: Nombre maximum de tiles (None = tous)

        Yields:
            ProcessedTile prêt pour inference V13
        """
        path = Path(path)
        input_type = self.detect_type(path)

        if input_type == InputType.PATCH:
            yield from self._process_patch(path)
        elif input_type == InputType.TILE:
            yield from self._process_tile(path)
        elif input_type == InputType.IMAGE:
            yield from self._process_image(path, max_tiles)
        elif input_type == InputType.WSI:
            yield from self._process_wsi(path, max_tiles)
        else:
            raise ValueError(f"Unknown input type for: {path}")

    def process_array(self,
                      images: np.ndarray,
                      source_name: str = "array") -> Generator[ProcessedTile, None, None]:
        """
        Traite un array numpy (ex: PanNuke .npy).

        Args:
            images: Array (N, 256, 256, 3) ou (256, 256, 3)
            source_name: Nom pour identifier la source

        Yields:
            ProcessedTile pour chaque image
        """
        if images.ndim == 3:
            images = images[np.newaxis, ...]

        for i, img in enumerate(images):
            if img.shape[:2] == (PANNUKE_SIZE, PANNUKE_SIZE):
                transformed = transform_pannuke_to_224(img, self.pannuke_method)
            elif img.shape[:2] == (TARGET_SIZE, TARGET_SIZE):
                transformed = img.copy()
            else:
                # Resize to 224×224
                transformed = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

            # Ensure uint8
            if transformed.dtype != np.uint8:
                if transformed.max() <= 1.0:
                    transformed = (transformed * 255).astype(np.uint8)
                else:
                    transformed = transformed.astype(np.uint8)

            yield ProcessedTile(
                image=transformed,
                x=0,
                y=0,
                source_path=Path(source_name),
                source_type=InputType.PATCH,
                metadata={"index": i}
            )

    def _process_patch(self, path: Path) -> Generator[ProcessedTile, None, None]:
        """Traite un PATCH (256×256)."""
        if path.suffix in ARRAY_EXTENSIONS:
            # Load numpy array
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                # NPZ file - assume 'images' key
                images = data.get('images', data[list(data.keys())[0]])
            else:
                images = data
            yield from self.process_array(images, str(path))
        else:
            # Load image file
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Cannot read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.shape[:2] == (PANNUKE_SIZE, PANNUKE_SIZE):
                transformed = transform_pannuke_to_224(img, self.pannuke_method)
            else:
                transformed = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

            yield ProcessedTile(
                image=transformed,
                x=0,
                y=0,
                source_path=path,
                source_type=InputType.PATCH
            )

    def _process_tile(self, path: Path) -> Generator[ProcessedTile, None, None]:
        """Traite un TILE (224×224) - direct sans transformation."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
            img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

        yield ProcessedTile(
            image=img,
            x=0,
            y=0,
            source_path=path,
            source_type=InputType.TILE
        )

    def _process_image(self, path: Path,
                       max_tiles: Optional[int] = None) -> Generator[ProcessedTile, None, None]:
        """Traite une IMAGE (<10000px) avec tiling simple."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        count = 0

        for y in range(0, h - TARGET_SIZE + 1, TARGET_SIZE):
            for x in range(0, w - TARGET_SIZE + 1, TARGET_SIZE):
                if max_tiles and count >= max_tiles:
                    return

                tile = img[y:y+TARGET_SIZE, x:x+TARGET_SIZE]

                # Filter if enabled
                if self.filter_tiles:
                    passed, reason = filter_tile(
                        tile,
                        check_tissue=True,
                        check_focus=True,
                        check_entropy=True
                    )
                    if not passed:
                        continue

                yield ProcessedTile(
                    image=tile.copy(),
                    x=x,
                    y=y,
                    source_path=path,
                    source_type=InputType.IMAGE
                )
                count += 1

    def _process_wsi(self, path: Path,
                     max_tiles: Optional[int] = None) -> Generator[ProcessedTile, None, None]:
        """
        Traite un WSI avec OpenSlide.

        Note: Cette méthode nécessite OpenSlide installé.
        Pour un traitement complet avec CLAM/HistoQC, voir le module
        src/wsi/wsi_processor.py (à implémenter en Phase 2).
        """
        try:
            import openslide
        except ImportError:
            raise ImportError(
                "OpenSlide is required for WSI processing. "
                "Install with: pip install openslide-python"
            )

        slide = openslide.OpenSlide(str(path))

        # Get dimensions at level 0
        w, h = slide.dimensions

        # Get MPP for proper scaling
        mpp = slide.properties.get('openslide.mpp-x')
        target_mpp = 0.5  # H-optimus standard

        # Calculate level to use (closest to target MPP)
        level = 0
        if mpp:
            mpp = float(mpp)
            for lvl in range(slide.level_count):
                lvl_mpp = mpp * slide.level_downsamples[lvl]
                if lvl_mpp >= target_mpp:
                    level = lvl
                    break

        # Get dimensions at selected level
        lvl_w, lvl_h = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]

        count = 0

        # Tile extraction at selected level
        for y in range(0, lvl_h - TARGET_SIZE + 1, TARGET_SIZE):
            for x in range(0, lvl_w - TARGET_SIZE + 1, TARGET_SIZE):
                if max_tiles and count >= max_tiles:
                    slide.close()
                    return

                # Read region (coordinates are at level 0)
                x0 = int(x * downsample)
                y0 = int(y * downsample)

                tile = slide.read_region((x0, y0), level, (TARGET_SIZE, TARGET_SIZE))
                tile = np.array(tile.convert('RGB'))

                # Filter if enabled
                if self.filter_tiles:
                    passed, reason = filter_tile(
                        tile,
                        check_tissue=True,
                        check_focus=True,
                        check_entropy=False  # Skip entropy for WSI (too slow)
                    )
                    if not passed:
                        continue

                yield ProcessedTile(
                    image=tile,
                    x=x0,
                    y=y0,
                    source_path=path,
                    source_type=InputType.WSI,
                    metadata={
                        "level": level,
                        "downsample": downsample,
                        "mpp": mpp
                    }
                )
                count += 1

        slide.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def process_pannuke_batch(
    images: np.ndarray,
    masks: Optional[np.ndarray] = None,
    method: str = "center_crop"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Traite un batch PanNuke complet (images + masques).

    Args:
        images: Array (N, 256, 256, 3)
        masks: Optional array (N, 256, 256) ou (N, 256, 256, C)
        method: "center_crop" ou "resize"

    Returns:
        (images_224, masks_224): Arrays transformés

    Example:
        >>> images = np.load("pannuke_images.npy")
        >>> masks = np.load("pannuke_masks.npy")
        >>> images_224, masks_224 = process_pannuke_batch(images, masks)
        >>> images_224.shape
        (N, 224, 224, 3)
    """
    n = images.shape[0]

    # Transform images
    images_224 = np.zeros((n, TARGET_SIZE, TARGET_SIZE, 3), dtype=images.dtype)
    for i in range(n):
        images_224[i] = transform_pannuke_to_224(images[i], method)

    # Transform masks if provided
    if masks is not None:
        if masks.ndim == 3:
            masks_224 = np.zeros((n, TARGET_SIZE, TARGET_SIZE), dtype=masks.dtype)
        else:
            masks_224 = np.zeros((n, TARGET_SIZE, TARGET_SIZE, masks.shape[-1]),
                                dtype=masks.dtype)
        for i in range(n):
            masks_224[i] = transform_masks_pannuke_to_224(masks[i], method)
        return images_224, masks_224

    return images_224, None
