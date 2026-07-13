import argparse
import cmath
import colorsys
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np

ESCAPE_RADIUS_SQ = 256.0**2
LOG2_LOG2_ESCAPE = math.log2(math.log2(256.0))
PALETTE_SIZE = 1024

try:
    import fractal_engine  # type: ignore[import-not-found]

    HAVE_ENGINE = True
except ImportError:
    fractal_engine = None
    HAVE_ENGINE = False


# --- fractal computation ---------------------------------------------------


def _compute_numpy(
    kind: str,
    center_x: float,
    center_y: float,
    zoom: float,
    width: int,
    height: int,
    max_iter: int,
    c: complex,
) -> np.ndarray:
    """Vectorized NumPy fallback mirroring the Rust engine.

    Works on separate real/imaginary arrays (not complex128) so the arithmetic
    matches the engine bit-for-bit; complex ops may fuse multiplies differently,
    which the chaotic iteration amplifies into visibly different pictures.
    """
    scale = 4.0 / (zoom * min(width, height))
    xs = (np.arange(width) - width / 2.0) * scale + center_x
    ys = (np.arange(height) - height / 2.0) * scale + center_y
    grid_r = np.tile(xs, height)
    grid_i = np.repeat(ys, width)

    if kind == "julia":
        zr, zi = grid_r, grid_i
        cr = np.full(zr.shape, c.real)
        ci = np.full(zi.shape, c.imag)
    else:
        zr = np.zeros(width * height)
        zi = np.zeros(width * height)
        cr, ci = grid_r, grid_i

    result = np.full(width * height, float(max_iter))
    alive = np.arange(width * height)

    for i in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        r2 = zr2 + zi2
        escaped = r2 > ESCAPE_RADIUS_SQ
        if escaped.any():
            result[alive[escaped]] = (
                i + 1.0 - np.log2(0.5 * np.log2(r2[escaped])) + LOG2_LOG2_ESCAPE
            )
            keep = ~escaped
            zr, zi, zr2, zi2 = zr[keep], zi[keep], zr2[keep], zi2[keep]
            cr, ci, alive = cr[keep], ci[keep], alive[keep]
            if alive.size == 0:
                break
        if kind == "burning_ship":
            zr = np.abs(zr)
            zi = np.abs(zi)
        elif kind == "tricorn":
            zi = -zi
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr

    return result.reshape(height, width)


def compute(
    kind: str,
    center: complex,
    zoom: float,
    width: int,
    height: int,
    max_iter: int,
    c: complex = 0j,
) -> np.ndarray:
    if HAVE_ENGINE:
        return fractal_engine.compute(
            kind,
            center.real,
            center.imag,
            zoom,
            width,
            height,
            max_iter,
            c.real,
            c.imag,
        )
    return _compute_numpy(
        kind, center.real, center.imag, zoom, width, height, max_iter, c
    )


def iter_for_zoom(zoom: float) -> int:
    return int(200 + 80 * math.log2(max(zoom, 1.0)))


# --- spots -------------------------------------------------------------------


@dataclass(frozen=True)
class Spot:
    kind: str
    center: complex
    zoom: float
    c: complex = 0j

    @property
    def max_iter(self) -> int:
        return iter_for_zoom(self.zoom)

    def compute(
        self,
        width: int,
        height: int,
        max_iter: int,
        zoom: float | None = None,
        c: complex | None = None,
    ) -> np.ndarray:
        return compute(
            self.kind,
            self.center,
            self.zoom if zoom is None else zoom,
            width,
            height,
            max_iter,
            self.c if c is None else c,
        )


def _cardioid_point(t: float) -> complex:
    return cmath.exp(1j * t) / 2 - cmath.exp(2j * t) / 4


def _random_offset(lo: float, hi: float) -> complex:
    return cmath.rect(random.uniform(lo, hi), random.uniform(0, 2 * math.pi))


def _mandelbrot_spot() -> Spot:
    if random.random() < 0.75:
        center = _cardioid_point(random.uniform(0, 2 * math.pi))
    else:
        center = -1.0 + cmath.rect(0.25, random.uniform(0, 2 * math.pi))
    center += _random_offset(0.001, 0.05)
    return Spot("mandelbrot", center, random.uniform(50, 5000))


def _julia_spot() -> Spot:
    if random.random() < 0.4:
        c = complex(-0.75 + random.uniform(-0.1, 0.1), random.uniform(-0.15, 0.15))
    else:
        c = _cardioid_point(random.uniform(0, 2 * math.pi))
    c += _random_offset(0.001, 0.03)
    return Spot("julia", 0j, random.uniform(0.8, 3.0), c)


def _burning_ship_spot() -> Spot:
    if random.random() < 0.6:
        center = complex(random.uniform(-1.8, -1.55), random.uniform(-0.08, 0.01))
        zoom = random.uniform(20, 2000)
    else:
        center = complex(random.uniform(-1.9, 0.8), random.uniform(-1.1, 0.3))
        zoom = random.uniform(3, 60)
    return Spot("burning_ship", center, zoom)


def _tricorn_spot() -> Spot:
    # On the real axis the tricorn equals the Mandelbrot set, so its antenna
    # band (dendrites and mini-copies) is a reliable source of detail; the
    # 3-fold rotational symmetry spreads the spot onto a random arm.
    base = complex(random.uniform(-1.99, -1.72), 0)
    arm = random.randrange(3) * 2 * math.pi / 3
    center = (base + _random_offset(0.001, 0.03)) * cmath.exp(1j * arm)
    return Spot("tricorn", center, random.uniform(30, 1500))


SPOT_GENERATORS = {
    "mandelbrot": _mandelbrot_spot,
    "julia": _julia_spot,
    "burning_ship": _burning_ship_spot,
    "tricorn": _tricorn_spot,
}
KIND_WEIGHTS = {"mandelbrot": 0.4, "julia": 0.25, "burning_ship": 0.2, "tricorn": 0.15}


def has_detail(spot: Spot, min_escape_frac: float = 0.05) -> bool:
    probe_size = 32
    iters = spot.compute(probe_size, probe_size, spot.max_iter)
    escaped = iters[iters < spot.max_iter]
    if escaped.size / (probe_size * probe_size) < min_escape_frac:
        return False
    return float(np.std(escaped)) > 1.0


def find_spot(kind: str | None, attempts: int = 12) -> Spot:
    spot = None
    for _ in range(attempts):
        picked = (
            kind
            or random.choices(list(KIND_WEIGHTS), weights=list(KIND_WEIGHTS.values()))[
                0
            ]
        )
        spot = SPOT_GENERATORS[picked]()
        if has_detail(spot):
            break
    assert spot is not None
    return spot


# --- coloring ----------------------------------------------------------------


def random_palette() -> np.ndarray:
    """Random gradient as a (PALETTE_SIZE, 3) uint8 lookup table."""
    n_keys = random.randint(4, 7)
    hue_start = random.random()
    hue_span = random.uniform(0.3, 0.8)

    positions = [0.0]
    colors = [(0.0, 0.0, 0.0)]
    for i in range(1, n_keys):
        hue = (hue_start + hue_span * (i - 1) / (n_keys - 2)) % 1.0
        sat = random.uniform(0.6, 1.0) if i < n_keys - 1 else random.uniform(0.1, 0.4)
        val = 0.3 + 0.7 * (i / (n_keys - 1))
        positions.append(i / (n_keys - 1))
        colors.append(colorsys.hsv_to_rgb(hue, sat, val))

    pos = np.array(positions)
    rgb = np.array(colors) * 255.0
    t = np.linspace(0.0, 1.0, PALETTE_SIZE)
    return np.stack(
        [np.interp(t, pos, rgb[:, channel]) for channel in range(3)], axis=1
    ).astype(np.uint8)


def colorize(
    iterations: np.ndarray,
    max_iter: int,
    lut: np.ndarray,
    vrange: tuple[float, float] | None = None,
) -> np.ndarray:
    height, width = iterations.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    outside = iterations < max_iter
    if not outside.any():
        return rgb

    smooth = iterations[outside]
    lo, hi = (
        vrange if vrange is not None else (float(smooth.min()), float(smooth.max()))
    )
    span = hi - lo
    if span > 0:
        normalized = np.clip((smooth - lo) / span, 0.0, 1.0)
    else:
        normalized = np.zeros_like(smooth)

    indices = normalized * (PALETTE_SIZE - 1)
    lower = np.clip(indices.astype(np.int64), 0, PALETTE_SIZE - 2)
    frac = (indices - lower)[:, np.newaxis]
    rgb[outside] = (lut[lower] * (1.0 - frac) + lut[lower + 1] * frac).astype(np.uint8)
    return rgb


def downsample_2x(rgb: np.ndarray) -> np.ndarray:
    """Gamma-correct 2x box downsample."""
    height, width = rgb.shape[0] // 2, rgb.shape[1] // 2
    linear = (rgb.astype(np.float64) / 255.0) ** 2.2
    averaged = linear.reshape(height, 2, width, 2, 3).mean(axis=(1, 3))
    return np.rint(averaged ** (1 / 2.2) * 255.0).astype(np.uint8)


# --- terminal output ---------------------------------------------------------


def get_render_size() -> tuple[int, int]:
    cols, rows = shutil.get_terminal_size()
    return cols, (rows - 1) * 2


def frame_string(rgb: np.ndarray) -> str:
    height, width = rgb.shape[:2]
    if height % 2 != 0:
        rgb = np.vstack([rgb, np.zeros((1, width, 3), dtype=np.uint8)])

    top_rows = rgb[0::2].tolist()
    bottom_rows = rgb[1::2].tolist()

    lines = []
    for top, bottom in zip(top_rows, bottom_rows, strict=True):
        parts = []
        prev_fg = prev_bg = None
        for fg, bg in zip(top, bottom, strict=True):
            fr, fg_, fb = fg
            br, bg_, bb = bg
            if fg != prev_fg and bg != prev_bg:
                parts.append(f"\033[38;2;{fr};{fg_};{fb};48;2;{br};{bg_};{bb}m▀")
            elif fg != prev_fg:
                parts.append(f"\033[38;2;{fr};{fg_};{fb}m▀")
            elif bg != prev_bg:
                parts.append(f"\033[48;2;{br};{bg_};{bb}m▀")
            else:
                parts.append("▀")
            prev_fg, prev_bg = fg, bg
        lines.append("".join(parts) + "\033[0m")

    return "\n".join(lines)


# --- rendering ---------------------------------------------------------------


def render_frame(
    spot: Spot,
    width: int,
    height: int,
    max_iter: int,
    lut: np.ndarray,
    zoom: float | None = None,
    c: complex | None = None,
    vrange: tuple[float, float] | None = None,
) -> tuple[str, tuple[float, float] | None]:
    """Render one 2x-supersampled frame; returns the ANSI string and the value
    range actually used (for smoothing across animation frames)."""
    iterations = spot.compute(width * 2, height * 2, max_iter, zoom=zoom, c=c)

    escaped = iterations[iterations < max_iter]
    if escaped.size:
        lo, hi = float(escaped.min()), float(escaped.max())
        if vrange is not None:
            lo = 0.7 * vrange[0] + 0.3 * lo
            hi = 0.7 * vrange[1] + 0.3 * hi
        vrange = (lo, hi)

    rgb = downsample_2x(colorize(iterations, max_iter, lut, vrange))
    return frame_string(rgb), vrange


def render_static(spot: Spot, width: int, height: int, lut: np.ndarray) -> None:
    frame, _ = render_frame(spot, width, height, spot.max_iter, lut)
    sys.stdout.write(frame + "\n")
    sys.stdout.flush()


def animate(
    spot: Spot, width: int, height: int, lut: np.ndarray, fps: float, duration: float
) -> tuple[int, float]:
    """Fly into the spot (or orbit c for Julia). Returns (frames, elapsed)."""
    n_frames = max(int(duration * fps), 2)
    zoom_start = min(0.7, spot.zoom)
    vrange = None
    rendered = 0

    sys.stdout.write("\033[?25l\033[2J")
    start = time.perf_counter()
    try:
        for i in range(n_frames):
            t = i / (n_frames - 1)
            eased = t * t * (3.0 - 2.0 * t)
            if spot.kind == "julia":
                zoom = spot.zoom
                c = spot.c + 0.015 * cmath.exp(2j * math.pi * eased)
                max_iter = spot.max_iter
            else:
                zoom = zoom_start * (spot.zoom / zoom_start) ** eased
                c = spot.c
                max_iter = iter_for_zoom(zoom)

            frame, vrange = render_frame(
                spot, width, height, max_iter, lut, zoom=zoom, c=c, vrange=vrange
            )
            sys.stdout.write("\033[H" + frame)
            sys.stdout.flush()
            rendered += 1

            deadline = start + (i + 1) / fps
            now = time.perf_counter()
            if now < deadline:
                time.sleep(deadline - now)
    finally:
        sys.stdout.write("\033[?25h\033[0m\n")
        sys.stdout.flush()

    return rendered, time.perf_counter() - start


# --- entry point -------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a random fractal in the terminal."
    )
    parser.add_argument(
        "--fractal",
        choices=["random", "mandelbrot", "julia", "burning-ship", "tricorn"],
        default="random",
        help="fractal type (default: weighted random)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed; reuse it to replay a render"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="fly into the spot instead of a static render (Julia: orbit c)",
    )
    parser.add_argument(
        "--duration", type=float, default=6.0, help="animation length in seconds"
    )
    parser.add_argument(
        "--fps", type=float, default=24.0, help="animation frame rate cap"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed)

    if not HAVE_ENGINE:
        print(
            "\033[33mfractal_engine not built, using slower NumPy fallback\033[0m",
            file=sys.stderr,
        )

    kind = None if args.fractal == "random" else args.fractal.replace("-", "_")
    spot = find_spot(kind)
    lut = random_palette()
    width, height = get_render_size()

    print(f"\033[36mrendering {spot.kind}...\033[0m", file=sys.stderr)
    start = time.perf_counter()
    if args.animate:
        frames, elapsed = animate(spot, width, height, lut, args.fps, args.duration)
        stats = f"{frames} frames  {frames / elapsed:.1f} fps"
    else:
        render_static(spot, width, height, lut)
        stats = f"{time.perf_counter() - start:.2f}s"

    print(
        f"  \033[35m{spot.kind}\033[0m  {width}x{height}  {stats}"
        f"  \033[2mseed {seed} (replay: --seed {seed})\033[0m",
        file=sys.stderr,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
