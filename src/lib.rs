use pyo3::prelude::*;
pub mod prelude;
pub mod types;
pub mod model;
pub mod operator;
pub mod tensor;
pub mod graph;
pub mod databuffer;
pub mod error;
// use crate::prelude::*;
// use prelude::*;
use model::Model;
use tensor::TensorF32;
use types::{OpType, ActiMode, AggrMode, PoolType, DataType, Initializer, InitializerType, ParamSyncType, Optimizer, OptimizerType, LossType, MetricsType};
use operator::PyOperator;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
    
}


/// A Python module implemented in Rust.
#[pymodule]
fn ufront(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<TensorF32>()?;
    m.add_class::<OpType>()?;
    m.add_class::<ActiMode>()?;
    m.add_class::<AggrMode>()?;
    m.add_class::<PoolType>()?;
    m.add_class::<DataType>()?;
    m.add_class::<Initializer>()?;
    m.add_class::<InitializerType>()?;
    m.add_class::<ParamSyncType>()?;
    m.add_class::<Optimizer>()?;
    m.add_class::<OptimizerType>()?;
    m.add_class::<LossType>()?;
    m.add_class::<MetricsType>()?;

    m.add_class::<PyOperator>()?;

    Ok(())
}

