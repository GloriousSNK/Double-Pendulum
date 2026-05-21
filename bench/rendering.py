from __future__ import annotations

import io
import base64
import numpy as np
from PIL import Image, ImageDraw

from .simulator import PendulumParams, bob_positions

def _hex_to_rgb(c: str):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))

def render_state(theta: np.ndarray, p: PendulumParams,
                 size=(512, 512), background: str = "#0a0a0a",
                 grid: bool = True) -> bytes:
    W, H = size
    img = Image.new("RGB", (W, H), _hex_to_rgb(background))
    draw = ImageDraw.Draw(img)

    reach = float(np.sum(p.L)) * 1.15
    cx, cy = W / 2, H / 2
    scale = min(W, H) / (2 * reach)

    def to_px(xy):
        return (cx + xy[0] * scale, cy - xy[1] * scale)

    if grid:
        gc = (30, 30, 30)
        step = scale
        x = cx % step
        while x < W:
            draw.line([(x, 0), (x, H)], fill=gc, width=1); x += step
        y = cy % step
        while y < H:
            draw.line([(0, y), (W, y)], fill=gc, width=1); y += step

    positions = bob_positions(theta, p)
    prev = (0.0, 0.0)
    rod_color = (200, 200, 200)
    bob_colors = [(58, 123, 213), (224, 90, 90), (80, 200, 120),
                  (245, 166, 35), (180, 79, 255), (0, 229, 204),
                  (255, 105, 180), (255, 215, 0)]
    for i, xy in enumerate(positions):
        draw.line([to_px(prev), to_px(xy)], fill=rod_color, width=3)
        bx, by = to_px(xy)
        r = 8 + 2 * (p.m[i] ** 0.5)
        col = bob_colors[i % len(bob_colors)]
        draw.ellipse([bx - r, by - r, bx + r, by + r], fill=col,
                     outline=(255, 255, 255), width=1)
        prev = (xy[0], xy[1])

    px, py = to_px((0, 0))
    draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def render_state_b64(theta, p, **kw) -> str:
    return base64.b64encode(render_state(theta, p, **kw)).decode("ascii")
