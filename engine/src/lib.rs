use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const ESCAPE_RADIUS_SQ: f64 = 256.0 * 256.0;
const LOG2_LOG2_ESCAPE: f64 = 3.0; // log2(log2(256))

#[inline]
fn mandelbrot_point(cr: f64, ci: f64, max_iter: u32) -> f64 {
    let mut zr = 0.0_f64;
    let mut zi = 0.0_f64;

    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > ESCAPE_RADIUS_SQ {
            let abs_z = (zr2 + zi2).sqrt();
            return i as f64 + 1.0 - abs_z.ln().log2() + LOG2_LOG2_ESCAPE;
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    max_iter as f64
}

#[inline]
fn julia_point(zr_start: f64, zi_start: f64, cr: f64, ci: f64, max_iter: u32) -> f64 {
    let mut zr = zr_start;
    let mut zi = zi_start;

    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > ESCAPE_RADIUS_SQ {
            let abs_z = (zr2 + zi2).sqrt();
            return i as f64 + 1.0 - abs_z.ln().log2() + LOG2_LOG2_ESCAPE;
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    max_iter as f64
}

#[pyfunction]
fn compute_mandelbrot<'py>(
    py: Python<'py>,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    width: usize,
    height: usize,
    max_iter: u32,
) -> Bound<'py, PyArray2<f64>> {
    let min_dim = width.min(height) as f64;
    let scale = 4.0 / (zoom * min_dim);
    let half_w = width as f64 / 2.0;
    let half_h = height as f64 / 2.0;

    let mut result = Array2::<f64>::zeros((height, width));
    result
        .rows_mut()
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>()
        .par_iter_mut()
        .for_each(|(row, row_data)| {
            let ci = (*row as f64 - half_h) * scale + center_y;
            for col in 0..width {
                let cr = (col as f64 - half_w) * scale + center_x;
                row_data[col] = mandelbrot_point(cr, ci, max_iter);
            }
        });

    result.into_pyarray(py)
}

#[pyfunction]
fn compute_julia<'py>(
    py: Python<'py>,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    width: usize,
    height: usize,
    max_iter: u32,
    c_real: f64,
    c_imag: f64,
) -> Bound<'py, PyArray2<f64>> {
    let min_dim = width.min(height) as f64;
    let scale = 4.0 / (zoom * min_dim);
    let half_w = width as f64 / 2.0;
    let half_h = height as f64 / 2.0;

    let mut result = Array2::<f64>::zeros((height, width));
    result
        .rows_mut()
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>()
        .par_iter_mut()
        .for_each(|(row, row_data)| {
            let zi_start = (*row as f64 - half_h) * scale + center_y;
            for col in 0..width {
                let zr_start = (col as f64 - half_w) * scale + center_x;
                row_data[col] = julia_point(zr_start, zi_start, c_real, c_imag, max_iter);
            }
        });

    result.into_pyarray(py)
}

#[pymodule]
fn fractal_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mandelbrot, m)?)?;
    m.add_function(wrap_pyfunction!(compute_julia, m)?)?;
    Ok(())
}
