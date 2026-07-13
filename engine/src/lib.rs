use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const ESCAPE_RADIUS_SQ: f64 = 256.0 * 256.0;
const LOG2_LOG2_ESCAPE: f64 = 3.0; // log2(log2(256)), keeps smoothing continuous at the escape radius

#[derive(Clone, Copy)]
enum Kind {
    Mandelbrot,
    Julia,
    BurningShip,
    Tricorn,
}

/// Smooth escape-time iteration of z -> pre(z)^2 + c.
///
/// All supported fractals share this loop and differ only in `pre`:
/// identity (Mandelbrot/Julia), |Re| + i|Im| (Burning Ship), conjugate (Tricorn).
#[inline]
fn escape_time(
    mut zr: f64,
    mut zi: f64,
    cr: f64,
    ci: f64,
    max_iter: u32,
    pre: impl Fn(f64, f64) -> (f64, f64),
) -> f64 {
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > ESCAPE_RADIUS_SQ {
            let log2_abs = 0.5 * (zr2 + zi2).log2();
            return f64::from(i) + 1.0 - log2_abs.log2() + LOG2_LOG2_ESCAPE;
        }
        let (tr, ti) = pre(zr, zi);
        zi = 2.0 * tr * ti + ci;
        zr = tr * tr - ti * ti + cr;
    }
    f64::from(max_iter)
}

fn render_grid(
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    point: impl Fn(f64, f64) -> f64 + Sync,
) -> Array2<f64> {
    if width == 0 || height == 0 {
        return Array2::zeros((height, width));
    }

    let scale = 4.0 / (zoom * width.min(height) as f64);
    let half_w = width as f64 / 2.0;
    let half_h = height as f64 / 2.0;

    let mut data = vec![0.0_f64; height * width];
    data.par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out_row)| {
            let y = (row as f64 - half_h) * scale + center_y;
            for (col, out) in out_row.iter_mut().enumerate() {
                let x = (col as f64 - half_w) * scale + center_x;
                *out = point(x, y);
            }
        });

    Array2::from_shape_vec((height, width), data).expect("vec length matches (height, width)")
}

#[pyfunction]
#[pyo3(signature = (kind, center_x, center_y, zoom, width, height, max_iter, c_real=0.0, c_imag=0.0))]
#[allow(clippy::too_many_arguments)]
fn compute<'py>(
    py: Python<'py>,
    kind: &str,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    width: usize,
    height: usize,
    max_iter: u32,
    c_real: f64,
    c_imag: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let kind = match kind {
        "mandelbrot" => Kind::Mandelbrot,
        "julia" => Kind::Julia,
        "burning_ship" => Kind::BurningShip,
        "tricorn" => Kind::Tricorn,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown fractal kind: {other:?}"
            )))
        }
    };

    let result = py.allow_threads(move || match kind {
        Kind::Mandelbrot => render_grid(width, height, center_x, center_y, zoom, |x, y| {
            escape_time(0.0, 0.0, x, y, max_iter, |zr, zi| (zr, zi))
        }),
        Kind::Julia => render_grid(width, height, center_x, center_y, zoom, |x, y| {
            escape_time(x, y, c_real, c_imag, max_iter, |zr, zi| (zr, zi))
        }),
        Kind::BurningShip => render_grid(width, height, center_x, center_y, zoom, |x, y| {
            escape_time(0.0, 0.0, x, y, max_iter, |zr, zi| (zr.abs(), zi.abs()))
        }),
        Kind::Tricorn => render_grid(width, height, center_x, center_y, zoom, |x, y| {
            escape_time(0.0, 0.0, x, y, max_iter, |zr, zi| (zr, -zi))
        }),
    });

    Ok(result.into_pyarray(py))
}

#[pymodule]
fn fractal_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute, m)?)?;
    Ok(())
}
