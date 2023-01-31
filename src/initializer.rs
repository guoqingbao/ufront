
use pyo3::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult};
use pyo3::pyclass;
use std::collections::HashMap;

#[pyclass]
pub struct Initializer {
    // #[pyo3(get, set)]
    // init_type: InitializerType,
    // #[pyo3(get, set)]
    // seed : u32,
    // #[pyo3(get, set)]
    // minv : u32,
    // #[pyo3(get, set)]
    // maxv : u32,
    #[pyo3(get, set)]
    pub params: HashMap<String, String>,
}

#[pymethods]
impl Initializer {
    #[new]
    pub fn new(
        // init_type: InitializerType,
        params: HashMap<String, String>,
    ) -> PyResult<PyClassInitializer<Self>> {
        println!("Initializer::new");
        let op = Initializer {
            // init_type: init_type,
            params: params,
        };
        Ok(PyClassInitializer::from(op))
    }
}