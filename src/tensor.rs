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
use numpy::ndarray::{ArrayD};
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
// use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use log::{info, warn, error};
use half::f16;
use half::bf16;

pub trait TensorTrait {
    fn get_dims(&self) -> usize;
    fn get_ndarray(&self);
    fn set_ndarray(&self);
    fn get_gradients(&self);
    fn get_owner(&self);
}

pub struct TensorU {
    pub shape: Vec<usize>,
    pub data_buffer: DataBuffer,
}

impl TensorTrait for TensorU {
    fn get_dims(&self) -> usize {
        0
    }

    fn get_ndarray(&self) {}

    fn set_ndarray(&self) {}

    fn get_gradients(&self) {}

    fn get_owner(&self) {}
}

#[pyclass]
pub struct Tensor {
    pub tensor: Option<TensorU>,
    pub name: String,
    pub dtype: DataType,
}

#[pymethods]
impl Tensor {
    #[new]
    #[pyo3(signature = (**kwds))]
    pub fn new(kwds: Option<&PyDict>) -> PyResult<PyClassInitializer<Self>> {
        info!("PyTensor::new");
        match kwds {
            Some(para) => {
                if para.contains("np_tensor").unwrap() && para.contains("dtype").unwrap() && para.contains("name").unwrap() {
                    let dtype = para.get_item("dtype").unwrap().extract::<DataType>().unwrap();
                    let name = para.get_item("name").unwrap().extract::<String>().unwrap();               
                    let mut tensor = Tensor {
                        tensor: None,
                        name,
                        dtype: dtype,
                    };

                    let np_tensor = para.get_item("np_tensor");
                    match np_tensor {
                        Some(v) => {
                            tensor.set_ndarray(v);
                        }
                        _ => {panic! {"Invalid tensor argument!"};}
                    }
                    Ok(PyClassInitializer::from(tensor))
                } else {
                    panic! {"Missing important arguments!"};
                }
            }
            _=> { panic! {"Missing important arguments!"}; }
        }
        
    }

    #[setter]
    pub fn set_ndarray(&mut self, arr: &PyAny) {
        let np_tensor = arr.extract::<PyReadonlyArrayDyn<f32>>();
        match np_tensor {
            Ok(v) => { 
                let x = v.as_array();
                match x.as_slice() {
                    Some(v) => {
                        self.tensor = Some(TensorU {
                            shape: x.shape().to_vec(),
                            data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(v.len(), Some(v.to_vec()))),
                        });
                        info!(
                            "Tensor initialized with {:?} dimension within Rust",
                            x.shape().to_vec()
                        );
                    }
                    _ => panic!("Invalid tensor inputs!"),
                }
            }
            _ => {
                let np_tensor16 = arr.extract::<PyReadonlyArrayDyn<f16>>();
                match np_tensor16 {
                    Ok(v) => { 
                        let x = v.as_array();
                        match x.as_slice() {
                            Some(v) => {
                                self.tensor = Some(TensorU {
                                    shape: x.shape().to_vec(),
                                    data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(v.len(), Some(v.to_vec()))),
                                });
                                info!(
                                    "Tensor initialized with {:?} dimension within Rust",
                                    x.shape().to_vec()
                                );
                            }
                            _ => panic!("Invalid tensor inputs!"),
                        }
                    }
                    _ => {
                        // panic!("Not supported dtype!");
                        let np_tensorb16 = arr.extract::<PyReadonlyArrayDyn<bf16>>();
                        match np_tensorb16 {
                            Ok(v) => { 
                                let x = v.as_array();
                                match x.as_slice() {
                                    Some(v) => {
                                        self.tensor = Some(TensorU {
                                            shape: x.shape().to_vec(),
                                            data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(v.len(), Some(v.to_vec()))),
                                        });
                                        info!(
                                            "Tensor initialized with {:?} dimension within Rust",
                                            x.shape().to_vec()
                                        );
                                    }
                                    _ => panic!("Invalid tensor inputs!"),
                                }
                            }
                            _=> {
                                let np_tensorf64 = arr.extract::<PyReadonlyArrayDyn<f64>>();
                                match np_tensorf64 {
                                    Ok(v) => { 
                                        let x = v.as_array();
                                        match x.as_slice() {
                                            Some(v) => {
                                                self.tensor = Some(TensorU {
                                                    shape: x.shape().to_vec(),
                                                    data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(v.len(), Some(v.to_vec()))),
                                                });
                                                info!(
                                                    "Tensor initialized with {:?} dimension within Rust",
                                                    x.shape().to_vec()
                                                );
                                            }
                                            _ => panic!("Invalid tensor inputs!"),
                                        }
                                    }
                                    _=> { 
                                        let np_tensori32 = arr.extract::<PyReadonlyArrayDyn<i32>>();
                                        match np_tensori32 {
                                            Ok(v) => { 
                                                let x = v.as_array();
                                                match x.as_slice() {
                                                    Some(v) => {
                                                        self.tensor = Some(TensorU {
                                                            shape: x.shape().to_vec(),
                                                            data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(v.len(), Some(v.to_vec()))),
                                                        });
                                                        info!(
                                                            "Tensor initialized with {:?} dimension within Rust",
                                                            x.shape().to_vec()
                                                        );
                                                    }
                                                    _ => panic!("Invalid tensor inputs!"),
                                                }
                                            }
                                            _=> { 
                                                let np_tensori64 = arr.extract::<PyReadonlyArrayDyn<i64>>();
                                                match np_tensori64 {
                                                    Ok(v) => { 
                                                        let x = v.as_array();
                                                        match x.as_slice() {
                                                            Some(v) => {
                                                                self.tensor = Some(TensorU {
                                                                    shape: x.shape().to_vec(),
                                                                    data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(v.len(), Some(v.to_vec()))),
                                                                });
                                                                info!(
                                                                    "Tensor initialized with {:?} dimension within Rust",
                                                                    x.shape().to_vec()
                                                                );
                                                            }
                                                            _ => panic!("Invalid tensor inputs!"),
                                                        }
                                                    }
                                                    _=> { 
                                                        let np_tensorbool = arr.extract::<PyReadonlyArrayDyn<bool>>();
                                                        match np_tensorbool {
                                                            Ok(v) => { 
                                                                let x = v.as_array();
                                                                match x.as_slice() {
                                                                    Some(v) => {
                                                                        self.tensor = Some(TensorU {
                                                                            shape: x.shape().to_vec(),
                                                                            data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(v.len(), Some(v.to_vec()))),
                                                                        });
                                                                        info!(
                                                                            "Tensor initialized with {:?} dimension within Rust",
                                                                            x.shape().to_vec()
                                                                        );
                                                                    }
                                                                    _ => panic!("Invalid tensor inputs!"),
                                                                }
                                                            }
                                                            _=> { 
                                                                panic!("Not supported tensor type!");
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                    }
                                }
                                
                            }
                        }
                    }
                }
                
            }
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
    pub fn get_ndarray<'py>(&mut self, py: Python<'py>) -> &'py PyAny {
        
        match &mut self.tensor {
            Some(v) => match &mut v.data_buffer {
                DataBuffer::CPUDataBuffer(data) => {
                    if self.dtype == DataType::Float {
                        ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec::<f32>())
                        .unwrap()
                        .into_pyarray(py)
                    } else if self.dtype == DataType::Half {
                        // let v32 = data.to_vec();
                        // let v16 = v32.into_iter().map(f16::from_f32).collect();
                        ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec::<f16>())
                        .unwrap()
                        .into_pyarray(py)
                    } 
                    else if self.dtype == DataType::BHalf {
                        ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec::<bf16>())
                        .unwrap()
                        .into_pyarray(py)
                    } 
                    else if self.dtype == DataType::Double {
                        ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec::<f64>())
                        .unwrap()
                        .into_pyarray(py)
                    } 
                    else if self.dtype == DataType::Int32 {
                        ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec::<i32>())
                        .unwrap()
                        .into_pyarray(py)
                    } 
                    else if self.dtype == DataType::Int64 {
                        ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec::<i64>())
                        .unwrap()
                        .into_pyarray(py)
                    } 
                    else if self.dtype == DataType::Bool {
                        ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec::<bool>())
                        .unwrap()
                        .into_pyarray(py)
                    } 
                    else {
                        panic!("Not supported tensor type!");
                    }

                }
                _ => panic!("Tensor conversion failed!"),
            },
            _ => {
                panic!("Tensor not initialized!");
            }

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
                if self.dtype == DataType::Float {
                    joined.extend(["f32".to_string()]);
                } else if self.dtype == DataType::Half {
                    joined.extend(["f16".to_string()]);
                } else if self.dtype == DataType::BHalf {
                    joined.extend(["bf16".to_string()]);
                } else if self.dtype == DataType::Double {
                    joined.extend(["f64".to_string()]);
                } else if self.dtype == DataType::Int32 {
                    joined.extend(["i32".to_string()]);
                } else if self.dtype == DataType::Int64 {
                    joined.extend(["i64".to_string()]);
                } else if self.dtype == DataType::Bool {
                    joined.extend(["bool".to_string()]);
                } else {
                    panic!("Not supported data type {:?}!", self.dtype);
                }
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
                if self.dtype == DataType::Float {
                    return v.shape.iter().copied().reduce(|a, b| a*b).unwrap() * 4;
                } else {
                    return v.shape.iter().copied().reduce(|a, b| a*b).unwrap() * 2;
                }
            }
            _ => panic!("Not initialized tensor!"),
        }
        // return 0;
    }
}
