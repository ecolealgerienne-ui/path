"""
CellViT-Optimus — Tile Server for WSI Deep Zoom Viewing.

Serves tiles from WSI files using OpenSlide's DeepZoomGenerator.
Compatible with OpenSeadragon viewer.

Usage:
    # Standalone
    python -m src.wsi.tile_server --wsi_dir data/wsi_test --port 8000

    # Programmatic
    from src.wsi.tile_server import create_tile_server_app, run_tile_server
    app = create_tile_server_app(wsi_dir)
    run_tile_server(app, port=8000)
"""

import os
import io
import logging
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache
import threading

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WSI Extensions supportées
WSI_EXTENSIONS = {'.svs', '.ndpi', '.mrxs', '.scn', '.tiff', '.tif', '.bif'}

# Cache pour les DeepZoomGenerator (évite de ré-ouvrir les slides)
_dz_cache: Dict[str, any] = {}
_dz_lock = threading.Lock()

# Tile format settings
TILE_SIZE = 254  # OpenSeadragon default
TILE_OVERLAP = 1
TILE_FORMAT = "jpeg"
TILE_QUALITY = 85


def get_deep_zoom_generator(slide_path: Path):
    """
    Get or create DeepZoomGenerator for a slide.

    Uses caching to avoid reopening slides.
    """
    import openslide
    from openslide.deepzoom import DeepZoomGenerator

    key = str(slide_path)

    with _dz_lock:
        if key not in _dz_cache:
            try:
                slide = openslide.OpenSlide(str(slide_path))
                dz = DeepZoomGenerator(
                    slide,
                    tile_size=TILE_SIZE,
                    overlap=TILE_OVERLAP,
                    limit_bounds=True
                )
                _dz_cache[key] = {
                    'slide': slide,
                    'dz': dz,
                    'path': slide_path,
                }
                logger.info(f"Opened slide: {slide_path.name}")
            except Exception as e:
                logger.error(f"Failed to open slide {slide_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Cannot open slide: {e}")

        return _dz_cache[key]['dz'], _dz_cache[key]['slide']


def close_all_slides():
    """Close all cached slides."""
    with _dz_lock:
        for key, data in _dz_cache.items():
            try:
                data['slide'].close()
            except Exception:
                pass
        _dz_cache.clear()
    logger.info("Closed all cached slides")


def create_tile_server_app(wsi_dir: str = "data/wsi_test") -> FastAPI:
    """
    Create FastAPI app for serving WSI tiles.

    Args:
        wsi_dir: Directory containing WSI files

    Returns:
        FastAPI application
    """
    wsi_path = Path(wsi_dir)

    app = FastAPI(
        title="CellViT-Optimus Tile Server",
        description="Serves Deep Zoom tiles for WSI viewing",
        version="1.0.0",
    )

    # Enable CORS for Gradio integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        """Health check endpoint."""
        return {"status": "ok", "service": "tile-server"}

    @app.get("/slides")
    def list_slides():
        """List available WSI files."""
        slides = []
        if wsi_path.exists():
            for ext in WSI_EXTENSIONS:
                slides.extend([f.name for f in wsi_path.glob(f"*{ext}")])
                slides.extend([f.name for f in wsi_path.glob(f"*{ext.upper()}")])
        return {"slides": sorted(set(slides))}

    @app.get("/slide/{slide_name}/info")
    def get_slide_info(slide_name: str):
        """Get slide metadata and Deep Zoom info."""
        slide_file = wsi_path / slide_name

        if not slide_file.exists():
            raise HTTPException(status_code=404, detail="Slide not found")

        try:
            dz, slide = get_deep_zoom_generator(slide_file)

            # Get properties
            props = dict(slide.properties)
            mpp_x = props.get('openslide.mpp-x', 'N/A')
            mpp_y = props.get('openslide.mpp-y', 'N/A')
            vendor = props.get('openslide.vendor', 'unknown')

            return {
                "name": slide_name,
                "width": slide.dimensions[0],
                "height": slide.dimensions[1],
                "mpp_x": mpp_x,
                "mpp_y": mpp_y,
                "vendor": vendor,
                "level_count": dz.level_count,
                "tile_size": TILE_SIZE,
                "overlap": TILE_OVERLAP,
                "format": TILE_FORMAT,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/slide/{slide_name}/dzi")
    def get_dzi(slide_name: str):
        """
        Get Deep Zoom Image descriptor (DZI XML).

        This is what OpenSeadragon needs to initialize the viewer.
        """
        slide_file = wsi_path / slide_name

        if not slide_file.exists():
            raise HTTPException(status_code=404, detail="Slide not found")

        try:
            dz, slide = get_deep_zoom_generator(slide_file)

            # Generate DZI XML
            dzi_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
    Format="{TILE_FORMAT}"
    Overlap="{TILE_OVERLAP}"
    TileSize="{TILE_SIZE}">
    <Size Width="{slide.dimensions[0]}" Height="{slide.dimensions[1]}"/>
</Image>'''

            return Response(
                content=dzi_xml,
                media_type="application/xml",
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/slide/{slide_name}/tiles/{level}/{col}_{row}.{format}")
    def get_tile(slide_name: str, level: int, col: int, row: int, format: str = "jpeg"):
        """
        Get a single tile from the slide.

        Args:
            slide_name: Name of the WSI file
            level: Zoom level (0 = most zoomed out)
            col: Column index (x)
            row: Row index (y)
            format: Image format (jpeg/png)
        """
        slide_file = wsi_path / slide_name

        if not slide_file.exists():
            raise HTTPException(status_code=404, detail="Slide not found")

        try:
            dz, _ = get_deep_zoom_generator(slide_file)

            # Validate level
            if level < 0 or level >= dz.level_count:
                raise HTTPException(status_code=400, detail=f"Invalid level: {level}")

            # Validate tile coordinates
            tiles_x, tiles_y = dz.level_tiles[level]
            if col < 0 or col >= tiles_x or row < 0 or row >= tiles_y:
                raise HTTPException(status_code=404, detail="Tile not found")

            # Get tile
            tile = dz.get_tile(level, (col, row))

            # Convert to bytes
            buf = io.BytesIO()
            if format.lower() == "png":
                tile.save(buf, "PNG")
                media_type = "image/png"
            else:
                tile.save(buf, "JPEG", quality=TILE_QUALITY)
                media_type = "image/jpeg"

            buf.seek(0)

            return Response(
                content=buf.read(),
                media_type=media_type,
                headers={
                    "Cache-Control": "public, max-age=86400",  # Cache 24h
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting tile: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/slide/{slide_name}/thumbnail")
    def get_thumbnail(slide_name: str, max_size: int = 512):
        """Get slide thumbnail."""
        slide_file = wsi_path / slide_name

        if not slide_file.exists():
            raise HTTPException(status_code=404, detail="Slide not found")

        try:
            _, slide = get_deep_zoom_generator(slide_file)

            w, h = slide.dimensions
            ratio = max_size / max(w, h)
            thumb_size = (int(w * ratio), int(h * ratio))

            thumbnail = slide.get_thumbnail(thumb_size)

            buf = io.BytesIO()
            thumbnail.save(buf, "JPEG", quality=90)
            buf.seek(0)

            return Response(
                content=buf.read(),
                media_type="image/jpeg",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.on_event("shutdown")
    def shutdown_event():
        """Cleanup on shutdown."""
        close_all_slides()

    return app


def run_tile_server(app: FastAPI = None, host: str = "0.0.0.0", port: int = 8000, wsi_dir: str = None):
    """
    Run the tile server.

    Args:
        app: FastAPI app (created if None)
        host: Server host
        port: Server port
        wsi_dir: WSI directory (only used if app is None)
    """
    if app is None:
        app = create_tile_server_app(wsi_dir or "data/wsi_test")

    uvicorn.run(app, host=host, port=port, log_level="info")


def run_tile_server_background(wsi_dir: str = "data/wsi_test", port: int = 8000) -> threading.Thread:
    """
    Run tile server in background thread.

    Returns:
        Thread object (already started)
    """
    import uvicorn

    app = create_tile_server_app(wsi_dir)

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    logger.info(f"Tile server started on port {port}")
    return thread


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CellViT-Optimus Tile Server")
    parser.add_argument("--wsi_dir", type=str, default="data/wsi_test",
                        help="Directory containing WSI files")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    args = parser.parse_args()

    logger.info(f"Starting tile server on {args.host}:{args.port}")
    logger.info(f"WSI directory: {args.wsi_dir}")

    app = create_tile_server_app(args.wsi_dir)
    run_tile_server(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
