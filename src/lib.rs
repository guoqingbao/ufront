// Copyright 2023, Enflame Tech. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
use pyo3::prelude::*;
pub mod databuffer;
pub mod error;
pub mod graph;
pub mod initializer;
pub mod model;
pub mod operator;
pub mod optimizer;
pub mod prelude;
pub mod tensor;
pub mod types;
use initializer::Initializer;
use model::Model;
use operator::PyOperator;
use optimizer::Optimizer;
use tensor::TensorF32;
use types::{ActiMode, AggrMode, DataType, LossType, MetricsType, OpType, ParamSyncType, PoolType};

/// A Python module implemented in Rust.
#[pymodule]
fn ufront(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Model>()?;
    m.add_class::<TensorF32>()?;
    m.add_class::<OpType>()?;
    m.add_class::<ActiMode>()?;
    m.add_class::<AggrMode>()?;
    m.add_class::<PoolType>()?;
    m.add_class::<DataType>()?;
    m.add_class::<Initializer>()?;
    // m.add_class::<InitializerType>()?;
    m.add_class::<ParamSyncType>()?;
    m.add_class::<Optimizer>()?;
    // m.add_class::<OptimizerType>()?;
    m.add_class::<LossType>()?;
    m.add_class::<MetricsType>()?;

    m.add_class::<PyOperator>()?;

    Ok(())
}
