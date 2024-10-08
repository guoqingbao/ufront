use pyo3::prelude::*;
use pyo3::pyclass;
use std::collections::HashMap;
use log::{info, warn, error};
#[pyclass]
#[derive(Debug, Clone)]
pub struct Optimizer {
    // #[pyo3(get, set)]
    // pub optim_type: OptimizerType,
    #[pyo3(get, set)]
    pub params: HashMap<String, String>,
}

#[pymethods]
impl Optimizer {
    #[new]
    pub fn new(
        // optim_type: OptimizerType,
        params: HashMap<String, String>,
    ) -> PyResult<PyClassInitializer<Self>> {
        info!("Optimizer::new");
        let op = Optimizer {
            // optim_type: optim_type,
            params,
        };
        Ok(PyClassInitializer::from(op))
    }
}
