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
use crate::databuffer::DataBuffer;
use crate::databuffer::Buffer;
use crate::types::DataType;
use ndarray;
use ndarray::ArrayView;
use ndarray::ArrayViewD;
use num::Signed;
use numpy::ndarray::{ArrayD, Zip};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
// use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use log::{info, warn, error};

pub trait TensorTrait {
    fn get_dims(&self) -> usize;
    fn get_ndarray(&self);
    fn set_ndarray(&self);
    fn get_gradients(&self);
    fn get_owner(&self);
}

pub struct Tensor<T: Clone + Signed> {
    pub shape: Vec<usize>,
    pub data_buffer: DataBuffer<T>,
}

impl<T: Clone + Signed> TensorTrait for Tensor<T> {
    fn get_dims(&self) -> usize {
        0
    }

    fn get_ndarray(&self) {}

    fn set_ndarray(&self) {}

    fn get_gradients(&self) {}

    fn get_owner(&self) {}
}

#[pyclass]
pub struct TensorF32 {
    pub tensor: Option<Tensor<f32>>,
    pub name: String,
    pub dtype: DataType,
}

#[pymethods]
impl TensorF32 {
    #[new]
    pub fn new(x: PyReadonlyArrayDyn<f32>, name: String) -> PyResult<PyClassInitializer<Self>> {
        info!("PyTensor::new");
        let mut tensor = TensorF32 {
            tensor: None,
            name,
            dtype: DataType::Float,
        };
        tensor.set_ndarray(x);
        Ok(PyClassInitializer::from(tensor))
    }

    #[setter]
    pub fn set_ndarray(&mut self, x: PyReadonlyArrayDyn<f32>) {
        let x = x.as_array();
        match x.as_slice() {
            Some(v) => {
                self.tensor = Some(Tensor::<f32> {
                    shape: x.shape().to_vec(),
                    data_buffer: DataBuffer::CPUDataBuffer(Buffer::<f32>::new(v.len(), Some(v.to_vec()))),
                });
                info!(
                    "Tensor initialized with {:?} dimension within Rust",
                    x.shape().to_vec()
                );
            }
            _ => panic!("Invalid tensor inputs!"),
        }
    }

    #[getter]
    pub fn get_dims(&self) -> usize {
        match &self.tensor {
            Some(v) => v.shape.len(),
            _ => panic!("Not initialized tensor!"),
        }
    }

    #[getter]
    pub fn get_ndarray<'py>(&self, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        match &self.tensor {
            Some(v) => match &v.data_buffer {
                DataBuffer::CPUDataBuffer(data) => {
                    ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec())
                        .unwrap()
                        .into_pyarray(py)
                }
                _ => panic!("Tensor conversion failed!"),
            },
            _ => panic!("Tensor not initialized!"),
        }
    }

    #[getter]
    pub fn get_gradients(&self) {}

    #[getter]
    pub fn get_owner(&self) {}

    #[getter]
    pub fn get_shape(&self, py: Python<'_>) -> Py<PyList> {
        match &self.tensor {
            Some(v) => {
                let lst = PyList::new(py, v.shape.clone());
                lst.into()
            }
            _ => panic!("Not initialized tensor!"),
        }
    }

    #[getter]
    pub fn get_ir(&self) -> String {
        match &self.tensor {
            Some(v) => {
                let mut joined: Vec<String> = v.shape.iter().map(|&dim| dim.to_string()).collect();
                joined.extend(["f32".to_string()]);
                format!("tensor<{}>", joined.join("x"))
            }
            _ => panic!("Not initialized tensor!"),
        }
    }

    #[getter]
    pub fn get_dtype(&self) -> DataType {
        self.dtype
    }

    #[getter]
    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn raw_size(&self) -> usize {
        match &self.tensor {
            Some(v) => {
                return v.shape.iter().copied().reduce(|a, b| a*b).unwrap() * 4;
            }
            _ => panic!("Not initialized tensor!"),
        }
        return 0;
    }
}
