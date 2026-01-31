#!/usr/bin/env python3
"""
CellViT-Optimus — Professional WSI Viewer with OpenSeadragon.

Deep zoom viewer with pan/zoom like hospital-grade pathology software.
Uses OpenSeadragon (JavaScript) + FastAPI tile server (Python).

Usage:
    python -m src.ui.app_pro_viewer
    python src/ui/app_pro_viewer.py --wsi_dir data/wsi_test --port 7863
"""

import gradio as gr
import numpy as np
from pathlib import Path
import logging
import time
import threading
import math
from typing import Optional, List, Tuple

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports projet
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wsi.tile_server import run_tile_server_background, WSI_EXTENSIONS

# ==============================================================================
# CONSTANTES
# ==============================================================================

DEFAULT_WSI_DIR = "data/wsi_test"
TILE_SERVER_PORT = 8000


# ==============================================================================
# OPENSEADRAGON HTML TEMPLATE
# ==============================================================================

def create_openseadragon_html(
    slide_name: str,
    tile_server_url: str,
    width: int = 0,
    height: int = 0,
    container_id: str = "osd-viewer"
) -> str:
    """
    Create HTML/JS for OpenSeadragon viewer.

    Args:
        slide_name: Name of the WSI file
        tile_server_url: URL of the tile server
        width: Slide width in pixels
        height: Slide height in pixels
        container_id: ID of the viewer container

    Returns:
        HTML string with embedded OpenSeadragon
    """
    if not slide_name:
        return """
        <div style="display: flex; align-items: center; justify-content: center;
                    height: 600px; background: #1a1a2e; color: #888; font-family: system-ui;">
            <div style="text-align: center;">
                <div style="font-size: 48px; margin-bottom: 16px;">🔬</div>
                <div style="font-size: 18px;">Sélectionnez une lame pour commencer</div>
            </div>
        </div>
        """

    tiles_url = f"{tile_server_url}/slide/{slide_name}/tiles"

    # Calculate number of levels for deep zoom pyramid
    max_dim = max(width, height) if width > 0 and height > 0 else 100000
    max_level = int(math.ceil(math.log2(max_dim))) + 1

    return f"""
    <div id="{container_id}" style="width: 100%; height: 600px; background: #1a1a2e;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>

    <script>
    (function() {{
        // Destroy existing viewer if any
        if (window.osdViewer) {{
            window.osdViewer.destroy();
            window.osdViewer = null;
        }}

        const slideWidth = {width};
        const slideHeight = {height};
        const tileSize = 254;
        const tileOverlap = 1;
        const tilesUrl = "{tiles_url}";

        // Custom tile source for our FastAPI server
        const customTileSource = {{
            width: slideWidth,
            height: slideHeight,
            tileSize: tileSize,
            tileOverlap: tileOverlap,
            minLevel: 0,
            maxLevel: {max_level},

            getTileUrl: function(level, x, y) {{
                return tilesUrl + "/" + level + "/" + x + "_" + y + ".jpeg";
            }}
        }};

        // Create viewer
        window.osdViewer = OpenSeadragon({{
            id: "{container_id}",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: customTileSource,

            // Navigator (mini-map)
            showNavigator: true,
            navigatorPosition: "BOTTOM_RIGHT",
            navigatorSizeRatio: 0.15,
            navigatorMaintainSizeRatio: true,
            navigatorAutoFade: false,

            // Zoom settings
            minZoomLevel: 0.1,
            maxZoomLevel: 40,
            defaultZoomLevel: 1,
            visibilityRatio: 0.5,
            constrainDuringPan: true,

            // Controls
            showZoomControl: true,
            showHomeControl: true,
            showFullPageControl: true,
            showRotationControl: false,

            // Performance
            immediateRender: true,
            imageLoaderLimit: 10,
            maxImageCacheCount: 500,

            // Animation
            animationTime: 0.3,
            blendTime: 0.1,
            springStiffness: 15,

            // Mouse gestures
            gestureSettingsMouse: {{
                scrollToZoom: true,
                clickToZoom: true,
                dblClickToZoom: true,
                flickEnabled: true
            }},

            // Touch gestures
            gestureSettingsTouch: {{
                scrollToZoom: false,
                clickToZoom: false,
                dblClickToZoom: true,
                pinchToZoom: true,
                flickEnabled: true
            }},

            // Debug
            debugMode: false
        }});

        // Event handlers
        window.osdViewer.addHandler('open', function() {{
            console.log("OpenSeadragon: Slide loaded successfully");
        }});

        window.osdViewer.addHandler('open-failed', function(event) {{
            console.error("OpenSeadragon: Failed to load slide", event);
        }});

        window.osdViewer.addHandler('tile-load-failed', function(event) {{
            console.warn("Tile load failed:", event.tile.url);
        }});

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (!window.osdViewer) return;

            // Ignore if typing in input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            switch(e.key) {{
                case '+':
                case '=':
                    window.osdViewer.viewport.zoomBy(1.5);
                    e.preventDefault();
                    break;
                case '-':
                    window.osdViewer.viewport.zoomBy(0.67);
                    e.preventDefault();
                    break;
                case 'Home':
                case 'h':
                case 'H':
                    window.osdViewer.viewport.goHome();
                    e.preventDefault();
                    break;
                case 'f':
                case 'F':
                    window.osdViewer.setFullScreen(!window.osdViewer.isFullPage());
                    e.preventDefault();
                    break;
            }}
        }});

        console.log("OpenSeadragon initialized:", slideWidth, "x", slideHeight, "pixels");
    }})();
    </script>

    <style>
        #{container_id} {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #333;
        }}
        #{container_id} .navigator {{
            border: 2px solid #4a9eff !important;
            border-radius: 4px;
            background: rgba(0,0,0,0.7) !important;
        }}
        #{container_id} .displayregion {{
            border: 2px solid #ff6b6b !important;
        }}
    </style>
    """


def create_viewer_with_annotations_html(
    slide_name: str,
    tile_server_url: str,
    annotations: List[dict] = None,
    container_id: str = "osd-viewer"
) -> str:
    """
    Create OpenSeadragon viewer with annotation overlays.

    Args:
        slide_name: Name of the WSI file
        tile_server_url: URL of the tile server
        annotations: List of annotation dicts with {x, y, width, height, score, label}
        container_id: ID of the viewer container

    Returns:
        HTML string
    """
    if not slide_name:
        return create_openseadragon_html(None, tile_server_url, container_id)

    tiles_url = f"{tile_server_url}/slide/{slide_name}/tiles/"
    dzi_url = f"{tile_server_url}/slide/{slide_name}/dzi"

    # Generate annotations JavaScript
    annotations_js = "[]"
    if annotations:
        import json
        annotations_js = json.dumps(annotations)

    return f"""
    <div id="{container_id}" style="width: 100%; height: 600px; background: #1a1a2e; position: relative;"></div>
    <svg id="annotation-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></svg>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>

    <script>
    (function() {{
        // Destroy existing viewer if any
        if (window.osdViewer) {{
            window.osdViewer.destroy();
        }}

        const annotations = {annotations_js};

        // Create viewer
        window.osdViewer = OpenSeadragon({{
            id: "{container_id}",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",

            tileSources: {{
                Image: {{
                    xmlns: "http://schemas.microsoft.com/deepzoom/2008",
                    Url: "{tiles_url}",
                    Format: "jpeg",
                    Overlap: "1",
                    TileSize: "254",
                    Size: {{ Width: "50000", Height: "50000" }}
                }}
            }},

            showNavigator: true,
            navigatorPosition: "BOTTOM_RIGHT",
            navigatorSizeRatio: 0.15,
            navigatorAutoFade: false,

            minZoomLevel: 0.1,
            maxZoomLevel: 40,
            defaultZoomLevel: 0.5,
            visibilityRatio: 0.5,
            constrainDuringPan: true,

            showZoomControl: true,
            showHomeControl: true,
            showFullPageControl: true,
            showRotationControl: true,

            immediateRender: true,
            imageLoaderLimit: 5,
            maxImageCacheCount: 200,

            animationTime: 0.5,
            blendTime: 0.1,
            springStiffness: 10,

            gestureSettingsMouse: {{
                scrollToZoom: true,
                clickToZoom: true,
                dblClickToZoom: true,
                flickEnabled: true
            }}
        }});

        // Draw annotations when viewport changes
        function updateAnnotations() {{
            if (!window.osdViewer || !annotations.length) return;

            const viewer = window.osdViewer;
            const container = viewer.container;
            const svg = document.getElementById('annotation-overlay');

            if (!svg || !container) return;

            // Position SVG over viewer
            const rect = container.getBoundingClientRect();
            svg.style.width = rect.width + 'px';
            svg.style.height = rect.height + 'px';

            // Clear previous annotations
            svg.innerHTML = '';

            // Draw each annotation
            annotations.forEach((ann, i) => {{
                const topLeft = viewer.viewport.imageToViewerElementCoordinates(
                    new OpenSeadragon.Point(ann.x, ann.y)
                );
                const bottomRight = viewer.viewport.imageToViewerElementCoordinates(
                    new OpenSeadragon.Point(ann.x + ann.width, ann.y + ann.height)
                );

                const w = bottomRight.x - topLeft.x;
                const h = bottomRight.y - topLeft.y;

                // Skip if too small or off-screen
                if (w < 2 || h < 2) return;
                if (topLeft.x > rect.width || topLeft.y > rect.height) return;
                if (bottomRight.x < 0 || bottomRight.y < 0) return;

                // Color based on score
                let color = '#22aa22';  // Green
                if (ann.score >= 0.30) color = '#ff3333';  // Red
                else if (ann.score >= 0.10) color = '#ffa500';  // Orange
                else if (ann.score >= 0.05) color = '#ffdd00';  // Yellow

                // Create rectangle
                const rectEl = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rectEl.setAttribute('x', topLeft.x);
                rectEl.setAttribute('y', topLeft.y);
                rectEl.setAttribute('width', w);
                rectEl.setAttribute('height', h);
                rectEl.setAttribute('fill', color);
                rectEl.setAttribute('fill-opacity', '0.2');
                rectEl.setAttribute('stroke', color);
                rectEl.setAttribute('stroke-width', '2');
                svg.appendChild(rectEl);

                // Add label if visible enough
                if (w > 30 && h > 20) {{
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', topLeft.x + 4);
                    text.setAttribute('y', topLeft.y + 14);
                    text.setAttribute('fill', 'white');
                    text.setAttribute('font-size', '12');
                    text.setAttribute('font-family', 'system-ui');
                    text.textContent = ann.label || ((ann.score * 100).toFixed(0) + '%');
                    svg.appendChild(text);
                }}
            }});
        }}

        // Update on viewport changes
        window.osdViewer.addHandler('animation', updateAnnotations);
        window.osdViewer.addHandler('animation-finish', updateAnnotations);
        window.osdViewer.addHandler('zoom', updateAnnotations);
        window.osdViewer.addHandler('pan', updateAnnotations);
        window.osdViewer.addHandler('resize', updateAnnotations);
        window.osdViewer.addHandler('open', updateAnnotations);

        console.log("OpenSeadragon viewer initialized with", annotations.length, "annotations");
    }})();
    </script>

    <style>
        #{container_id} {{ border-radius: 8px; overflow: hidden; }}
        #annotation-overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; z-index: 100; }}
        .navigator {{ border: 2px solid #4a9eff !important; border-radius: 4px; }}
    </style>
    """


# ==============================================================================
# GRADIO APPLICATION
# ==============================================================================

class ProViewerState:
    """State for the professional viewer."""

    def __init__(self, wsi_dir: str = DEFAULT_WSI_DIR, tile_server_port: int = TILE_SERVER_PORT):
        self.wsi_dir = Path(wsi_dir)
        self.tile_server_port = tile_server_port
        self.tile_server_url = f"http://localhost:{tile_server_port}"
        self.available_files: List[str] = []
        self.selected_file: Optional[str] = None
        self.tile_server_thread: Optional[threading.Thread] = None
        self.annotations: List[dict] = []

    def scan_folder(self) -> List[str]:
        """Scan WSI directory."""
        self.available_files = []
        if self.wsi_dir.exists():
            for ext in WSI_EXTENSIONS:
                self.available_files.extend([f.name for f in self.wsi_dir.glob(f"*{ext}")])
                self.available_files.extend([f.name for f in self.wsi_dir.glob(f"*{ext.upper()}")])
        self.available_files = sorted(set(self.available_files))
        return self.available_files

    def start_tile_server(self):
        """Start tile server in background."""
        if self.tile_server_thread is None or not self.tile_server_thread.is_alive():
            self.tile_server_thread = run_tile_server_background(
                str(self.wsi_dir),
                self.tile_server_port
            )
            # Wait for server to be ready
            time.sleep(2)
            # Verify server is responding
            import requests
            for i in range(5):
                try:
                    resp = requests.get(f"{self.tile_server_url}/", timeout=2)
                    if resp.ok:
                        logger.info(f"Tile server running at {self.tile_server_url}")
                        return
                except Exception:
                    pass
                time.sleep(1)
            logger.warning("Tile server may not be fully ready")


# Global state
viewer_state = ProViewerState()


def on_slide_select(slide_name: str) -> Tuple[str, str]:
    """Handle slide selection."""
    logger.info(f"on_slide_select called with: {slide_name}")

    if not slide_name:
        return create_openseadragon_html(None, viewer_state.tile_server_url), "Sélectionnez une lame"

    viewer_state.selected_file = slide_name

    # Get slide info from tile server
    width, height = 0, 0
    info_text = f"### {slide_name}\n\n*Chargement...*"

    try:
        import requests
        url = f"{viewer_state.tile_server_url}/slide/{slide_name}/info"
        logger.info(f"Requesting slide info from: {url}")
        resp = requests.get(url, timeout=10)
        if resp.ok:
            info = resp.json()
            width = info.get('width', 0)
            height = info.get('height', 0)
            logger.info(f"Slide info received: {width}x{height}, levels={info.get('level_count')}")
            info_text = f"""### {slide_name}

**Dimensions:** {width:,} x {height:,} px
**MPP:** {info.get('mpp_x', 'N/A')}
**Scanner:** {info.get('vendor', 'N/A')}
**Niveaux:** {info.get('level_count', 'N/A')}"""
        else:
            logger.error(f"Tile server returned {resp.status_code}: {resp.text}")
            info_text = f"### {slide_name}\n\n*Erreur serveur: {resp.status_code}*"
    except requests.exceptions.ConnectionError:
        logger.error("Tile server not responding - attempting restart")
        viewer_state.start_tile_server()
        time.sleep(2)
        # Retry once
        try:
            resp = requests.get(f"{viewer_state.tile_server_url}/slide/{slide_name}/info", timeout=10)
            if resp.ok:
                info = resp.json()
                width = info.get('width', 0)
                height = info.get('height', 0)
                info_text = f"""### {slide_name}

**Dimensions:** {width:,} x {height:,} px
**MPP:** {info.get('mpp_x', 'N/A')}
**Scanner:** {info.get('vendor', 'N/A')}
**Niveaux:** {info.get('level_count', 'N/A')}"""
        except Exception as e2:
            info_text = f"### {slide_name}\n\n*Erreur connexion serveur tiles*"
    except Exception as e:
        logger.error(f"Error getting slide info: {e}")
        info_text = f"### {slide_name}\n\n*Erreur: {e}*"

    # Create viewer HTML with actual dimensions
    logger.info(f"Creating OpenSeadragon viewer: {slide_name}, {width}x{height}")
    viewer_html = create_openseadragon_html(
        slide_name=slide_name,
        tile_server_url=viewer_state.tile_server_url,
        width=width,
        height=height,
    )

    logger.info(f"Viewer HTML length: {len(viewer_html)} chars")
    return viewer_html, info_text


def create_pro_viewer_ui(wsi_dir: str = DEFAULT_WSI_DIR, tile_server_port: int = TILE_SERVER_PORT):
    """Create the professional viewer interface."""

    viewer_state.wsi_dir = Path(wsi_dir)
    viewer_state.tile_server_port = tile_server_port
    viewer_state.tile_server_url = f"http://localhost:{tile_server_port}"
    viewer_state.scan_folder()
    viewer_state.start_tile_server()

    custom_css = """
    .viewer-container { min-height: 620px; }
    .slide-info { font-family: system-ui; }
    """

    with gr.Blocks(
        title="CellViT-Optimus Pro Viewer",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:

        gr.Markdown("""
        # 🔬 CellViT-Optimus — Viewer Professionnel

        *Navigation fluide avec zoom/pan comme les logiciels hospitaliers*

        **Contrôles:**
        - 🖱️ **Scroll** = Zoom
        - 🖱️ **Drag** = Pan
        - ⌨️ **+/-** = Zoom in/out
        - ⌨️ **H** = Home (vue complète)
        - ⌨️ **F** = Plein écran
        - 🖱️ **Double-clic** = Zoom x2
        """)

        with gr.Row():
            # Left panel - Controls
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Lame")

                slide_dropdown = gr.Dropdown(
                    choices=viewer_state.available_files,
                    value=viewer_state.available_files[0] if viewer_state.available_files else None,
                    label="Fichier WSI",
                    interactive=True,
                )

                slide_info = gr.Markdown("*Sélectionnez une lame*")

                gr.Markdown("---")

                gr.Markdown("""
                ### 📊 Légende
                - 🔴 **Rouge** = Suspect (≥30%)
                - 🟠 **Orange** = À surveiller (≥10%)
                - 🟡 **Jaune** = Faible risque (≥5%)
                - 🟢 **Vert** = Normal
                """)

            # Center - Viewer
            with gr.Column(scale=4):
                # Start with placeholder, app.load() will populate
                viewer_html = gr.HTML(
                    value=create_openseadragon_html(None, viewer_state.tile_server_url),
                    elem_classes=["viewer-container"],
                )

        # Events
        slide_dropdown.change(
            fn=on_slide_select,
            inputs=[slide_dropdown],
            outputs=[viewer_html, slide_info],
        )

        # Load initial slide
        app.load(
            fn=on_slide_select,
            inputs=[slide_dropdown],
            outputs=[viewer_html, slide_info],
        )

    return app


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CellViT-Optimus Pro Viewer")
    parser.add_argument("--wsi_dir", type=str, default=DEFAULT_WSI_DIR,
                        help=f"WSI directory (default: {DEFAULT_WSI_DIR})")
    parser.add_argument("--port", type=int, default=7863,
                        help="Gradio port (default: 7863)")
    parser.add_argument("--tile_port", type=int, default=8000,
                        help="Tile server port (default: 8000)")
    parser.add_argument("--share", action="store_true",
                        help="Create public link")
    args = parser.parse_args()

    logger.info(f"Starting CellViT-Optimus Pro Viewer")
    logger.info(f"WSI directory: {args.wsi_dir}")
    logger.info(f"Gradio port: {args.port}")
    logger.info(f"Tile server port: {args.tile_port}")

    app = create_pro_viewer_ui(
        wsi_dir=args.wsi_dir,
        tile_server_port=args.tile_port
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
