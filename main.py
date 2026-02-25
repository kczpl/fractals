import colorsys
import math
import random
import shutil
import sys
import time

import numpy as np

try:
    from fractal_engine import compute_mandelbrot, compute_julia
except ImportError:
    _ESCAPE_RADIUS_SQ = 256.0 ** 2
    _LOG2_LOG2_ESCAPE = math.log2(math.log2(256.0))

    def compute_mandelbrot(center_x, center_y, zoom, width, height, max_iter):
        scale = 4.0 / (zoom * min(width, height))
        result = np.zeros((height, width), dtype=np.float64)
        for row in range(height):
            ci = (row - height / 2.0) * scale + center_y
            for col in range(width):
                cr = (col - width / 2.0) * scale + center_x
                zr, zi = 0.0, 0.0
                for iteration in range(max_iter):
                    zr2, zi2 = zr * zr, zi * zi
                    if zr2 + zi2 > _ESCAPE_RADIUS_SQ:
                        abs_z = math.sqrt(zr2 + zi2)
                        result[row, col] = iteration + 1.0 - math.log2(math.log2(abs_z)) + _LOG2_LOG2_ESCAPE
                        break
                    zi = 2.0 * zr * zi + ci
                    zr = zr2 - zi2 + cr
                else:
                    result[row, col] = float(max_iter)
        return result

    def compute_julia(center_x, center_y, zoom, width, height, max_iter, c_real, c_imag):
        scale = 4.0 / (zoom * min(width, height))
        result = np.zeros((height, width), dtype=np.float64)
        for row in range(height):
            for col in range(width):
                zr = (col - width / 2.0) * scale + center_x
                zi = (row - height / 2.0) * scale + center_y
                for iteration in range(max_iter):
                    zr2, zi2 = zr * zr, zi * zi
                    if zr2 + zi2 > _ESCAPE_RADIUS_SQ:
                        abs_z = math.sqrt(zr2 + zi2)
                        result[row, col] = iteration + 1.0 - math.log2(math.log2(abs_z)) + _LOG2_LOG2_ESCAPE
                        break
                    zi = 2.0 * zr * zi + c_imag
                    zr = zr2 - zi2 + c_real
                else:
                    result[row, col] = float(max_iter)
        return result


def generate_palette():
    n_keys = random.randint(4, 7)
    hue_start = random.random()
    hue_span = random.uniform(0.3, 0.8)

    keyframes = [(0.0, (0, 0, 0))]
    for i in range(1, n_keys):
        pos = i / (n_keys - 1)
        hue = (hue_start + hue_span * (i - 1) / (n_keys - 2)) % 1.0
        sat = random.uniform(0.6, 1.0) if i < n_keys - 1 else random.uniform(0.1, 0.4)
        val = 0.3 + 0.7 * (i / (n_keys - 1))
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        keyframes.append((pos, (int(r * 255), int(g * 255), int(b * 255))))
    return keyframes


def apply_palette(iterations, max_iter, keyframes):
    positions = np.array([kf[0] for kf in keyframes])
    colors = np.array([kf[1] for kf in keyframes])

    t = np.linspace(0.0, 1.0, 1024)
    lut = np.stack([
        np.interp(t, positions, colors[:, 0]),
        np.interp(t, positions, colors[:, 1]),
        np.interp(t, positions, colors[:, 2]),
    ], axis=1).astype(np.uint8)

    height, width = iterations.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    outside = iterations < max_iter
    if outside.any():
        smooth = iterations[outside]
        lo, hi = smooth.min(), smooth.max()
        normalized = (smooth - lo) / (hi - lo) if hi - lo > 0 else np.zeros_like(smooth)
        indices = normalized * 1023
        lower = np.clip(indices.astype(np.int64), 0, 1022)
        frac = (indices - lower)[:, np.newaxis]
        rgb[outside] = (lut[lower] * (1.0 - frac) + lut[lower + 1] * frac).astype(np.uint8)

    return rgb


def get_render_size():
    cols, rows = shutil.get_terminal_size()
    return cols, (rows - 1) * 2


def render_to_terminal(rgb):
    height, width = rgb.shape[:2]

    if height % 2 != 0:
        rgb = np.vstack([rgb, np.zeros((1, width, 3), dtype=np.uint8)])
        height += 1

    lines = []
    for y in range(0, height, 2):
        parts = []
        prev_fg = prev_bg = None
        for x in range(width):
            fg = (int(rgb[y, x, 0]), int(rgb[y, x, 1]), int(rgb[y, x, 2]))
            bg = (int(rgb[y + 1, x, 0]), int(rgb[y + 1, x, 1]), int(rgb[y + 1, x, 2]))
            esc = ""
            if fg != prev_fg:
                esc += f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m"
            if bg != prev_bg:
                esc += f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m"
            parts.append(esc + "\u2580")
            prev_fg, prev_bg = fg, bg
        lines.append("".join(parts) + "\033[0m")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def generate_mandelbrot_spot():
    t = random.uniform(0, 2 * math.pi)
    if random.random() < 0.75:
        cr = math.cos(t) / 2 - math.cos(2 * t) / 4
        ci = math.sin(t) / 2 - math.sin(2 * t) / 4
    else:
        angle = random.uniform(0, 2 * math.pi)
        cr = -1.0 + 0.25 * math.cos(angle)
        ci = 0.25 * math.sin(angle)

    perturb = random.uniform(0.001, 0.05)
    pr_angle = random.uniform(0, 2 * math.pi)
    cr += perturb * math.cos(pr_angle)
    ci += perturb * math.sin(pr_angle)
    zoom = random.uniform(50, 5000)

    return {"fractal": "mandelbrot", "center": (cr, ci), "zoom": zoom}


def generate_julia_spot():
    t = random.uniform(0, 2 * math.pi)
    if random.random() < 0.4:
        cr = -0.75 + random.uniform(-0.1, 0.1)
        ci = random.uniform(-0.15, 0.15)
    else:
        cr = math.cos(t) / 2 - math.cos(2 * t) / 4
        ci = math.sin(t) / 2 - math.sin(2 * t) / 4

    perturb = random.uniform(0.001, 0.03)
    pr_angle = random.uniform(0, 2 * math.pi)
    cr += perturb * math.cos(pr_angle)
    ci += perturb * math.sin(pr_angle)
    zoom = random.uniform(0.8, 3.0)

    return {"fractal": "julia", "center": (0.0, 0.0), "zoom": zoom, "c": (cr, ci)}


def has_detail(spot, max_iter, min_escape_frac=0.05):
    probe_size = 32
    center = spot["center"]
    zoom = spot["zoom"]
    if spot["fractal"] == "mandelbrot":
        iters = compute_mandelbrot(center[0], center[1], zoom, probe_size, probe_size, max_iter)
    else:
        c = spot["c"]
        iters = compute_julia(center[0], center[1], zoom, probe_size, probe_size, max_iter, c[0], c[1])
    escaped = iters[iters < max_iter]
    if len(escaped) / (probe_size * probe_size) < min_escape_frac:
        return False
    return np.std(escaped) > 1.0


def main():
    spot = generate_mandelbrot_spot() if random.random() < 0.65 else generate_julia_spot()
    for _ in range(10):
        probe_max_iter = int(200 + 80 * math.log2(max(spot["zoom"], 1.0)))
        if has_detail(spot, probe_max_iter):
            break
        spot = generate_mandelbrot_spot() if random.random() < 0.65 else generate_julia_spot()

    keyframes = generate_palette()
    center = spot["center"]
    zoom = spot["zoom"]
    width, height = get_render_size()
    max_iter = int(200 + 80 * math.log2(max(zoom, 1.0)))

    print(f"\033[36mrendering {spot['fractal']}...\033[0m", file=sys.stderr)
    start = time.perf_counter()

    ss_w, ss_h = width * 2, height * 2
    if spot["fractal"] == "mandelbrot":
        iterations = compute_mandelbrot(center[0], center[1], zoom, ss_w, ss_h, max_iter)
    else:
        c = spot["c"]
        iterations = compute_julia(center[0], center[1], zoom, ss_w, ss_h, max_iter, c[0], c[1])

    rgb_hi = apply_palette(iterations, max_iter, keyframes)
    rgb = rgb_hi.reshape(height, 2, width, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
    elapsed = time.perf_counter() - start

    render_to_terminal(rgb)
    print(f"  \033[35m{spot['fractal']}\033[0m  {width}x{height}  {elapsed:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
