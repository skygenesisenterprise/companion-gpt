use pyo3::prelude::*;
use numpy::{PyArray2, PyArray1, PyArrayMethods};

#[pyfunction]
fn sigmoid_rust(x: &PyArray1<f64>) -> PyResult<PyObject> {
    let x_array = unsafe { x.as_array() };
    let result: Vec<f64> = x_array.iter().map(|&val| 1.0 / (1.0 + (-val).exp())).collect();
    Ok(PyArray1::from_vec(x.py(), result).to_object(x.py()))
}

#[pyfunction]
fn dot_product_rust(a: &PyArray2<f64>, b: &PyArray2<f64>) -> PyResult<PyObject> {
    let a_array = unsafe { a.as_array() };
    let b_array = unsafe { b.as_array() };
    let result = a_array.dot(&b_array);
    Ok(PyArray2::from_array(a.py(), &result).to_object(a.py()))
}

#[pymodule]
fn companion_gpt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sigmoid_rust, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product_rust, m)?)?;
    Ok(())
}
