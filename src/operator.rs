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
// use crate::tensor::Tensor;
use crate::tensor::Tensor;
use crate::tensor::TensorU;
use crate::types::DataType;
use crate::types::OpType;
use core::panic;
use std::any::TypeId;
use half::{f16, bf16};
use ::num::abs;
use numpy::{PyReadonlyArrayDyn};
// use pyo3::exceptions::PyOSError;
use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
// use std::os::raw::c_void;
use itertools::Itertools;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
static IDX: AtomicUsize = AtomicUsize::new(0);
use log::{info, warn, error};
pub trait Texualization {
    fn dump(&self) -> String;
}

pub trait OperatorTrait {
    fn forward(&self);
    fn dump(&self) -> &str;
}
#[repr(C)]
pub struct Operator {
    // pub id : usize,
    pub inputs: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
    pub op_type: OpType,
    pub params: IndexMap<String, String>,
    pub channel_first: bool,
}

impl PartialEq for Operator {
    fn eq(&self, other: &Self) -> bool {
        self as *const Operator as u64 == other as *const Operator as u64
    }
}

// impl Clone for Operator {
//     fn clone(&self) -> Self {
//         let params_ = self.params.clone();
//         Operator {id : self.id, op_type : self.op_type, params : params_, inputs : self.inputs.clone(), outputs : self.outputs.clone()}
//     }
// }

impl OperatorTrait for Operator {
    fn forward(&self) {}

    fn dump(&self) -> &str {
        ""
    }
}

impl Texualization for Operator {
    fn dump(&self) -> String {
        "".to_string()
    }
}

impl Operator {
    pub fn new(op: OpType, params: IndexMap<String, String>) -> Operator {
        info!("Operator::new------------------------------------------------------------------");
        // let mut params_ : HashMap<String, String> = HashMap::new();
        // params_.extend(params.into_iter());
        // Python::with_gil(|py| {
        //     let inputs: Py<Tensor> = Py::new(py, Tensor { tensor : None})?;
        //     let outputs: Py<Tensor> = Py::new(py, Tensor { tensor : None})?;

        //     let operator = Operator {id : id, inputs : inputs, outputs : outputs, op_type : op, params : params_};
        //     // return operator;
        //     Ok(PyClassInitializer::from(operator))
        // })
        Operator {
            inputs: Vec::new(),
            outputs: Vec::new(),
            op_type: op,
            params,
            channel_first: true,
        }
    }

    pub fn num_of_inputs(&self) -> usize {
        //
        let num = self.inputs.len();
        info!("Operator::num_of_inputs {num}");
        num
    }

    pub fn num_of_outputs(&self) -> usize {
        let num = self.outputs.len();
        info!("Operator::num_of_outputs {num}");
        num
    }

    pub fn empty_tensor(typeid: TypeId, sz: usize, shape: Vec<usize>) -> TensorU {
        if typeid == TypeId::of::<f32>() {
            TensorU {
                shape: shape,
                data_buffer: DataBuffer::CPUDataBuffer(Buffer::new::<f32>(sz, None)),
            }
        } else if typeid == TypeId::of::<f16>() {
            TensorU {
                shape: shape,
                data_buffer: DataBuffer::CPUDataBuffer(Buffer::new::<f16>(sz, None)),
            }
        } else if typeid == TypeId::of::<bf16>() {
            TensorU {
                shape: shape,
                data_buffer: DataBuffer::CPUDataBuffer(Buffer::new::<bf16>(sz, None)),
            }
        } else if typeid == TypeId::of::<i32>() {
            TensorU {
                shape: shape,
                data_buffer: DataBuffer::CPUDataBuffer(Buffer::new::<i32>(sz, None)),
            }
        } else {
            panic!("Invalid type! {:?}", typeid);
        }
    }
    // pub fn get_input_ndarray<'py>(&self, idx : usize, py: Python<'py>) -> &'py PyArrayDyn<f32>{
    //     info!("Operator::get_input_ndarray {}", idx);
    //     if idx < self.num_of_inputs() {

    //         match &self.inputs[idx].tensor {
    //             Some(v) => {
    //                 match &v.data_buffer {
    //                     DataBuffer::CPUDataBuffer(data) => ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec()).unwrap().into_pyarray(py),
    //                     _ => panic!("Tensor conversion failed!")
    //                 }
    //             }
    //             _ => panic!("Tensor not initialized!")
    //         }
    //     } else {
    //         panic!("Tensor not initialized!")
    //     }

    // }

    // pub fn get_input(&self, idx : usize) -> Result<&Tensor, ()> {

    //     Err(())
    //     // return Err(PyOSError::new_err("Failed to obtain tensor!".to_string()))
    // }

    pub fn get_input<'py>(&mut self, idx: usize, py: Python<'py>) -> Py<Tensor> {
        info!("Operator::get_input idx {idx}");
        if idx < self.num_of_inputs() {
            // return Ok(&mut self.inputs[idx]);
            // return self.inputs[idx];
            match &self.inputs[idx].tensor {
                Some(v) => {
                    let tensor_ = TensorU {
                        shape: v.shape.to_vec(),
                        data_buffer: v.data_buffer.clone(),
                    };

                    return Py::new(
                            py,
                            Tensor {
                                tensor: Some(tensor_),
                                name: self.inputs[idx].name.clone(),
                                dtype: self.inputs[idx].dtype,
                            },
                        )
                        .unwrap();
                }
                _ => {
                    panic!("Unable to obtain output!");
                }

            }
        }
        panic!("Error get_input!");
        // Err(())
    }

    pub fn get_output<'py>(&self, idx: usize, py: Python<'py>) -> Py<Tensor>{
        info!("Operator::get_output idx {idx}");
        if idx < self.num_of_outputs() {
            // return Ok(&self.outputs[idx]);
            match &self.outputs[idx].tensor {
                Some(v) => {
                    let tensor_ = TensorU {
                        shape: v.shape.to_vec(),
                        data_buffer: v.data_buffer.clone(),
                    };

                    return Py::new(
                            py,
                            Tensor {
                                tensor: Some(tensor_),
                                name: self.outputs[idx].name.clone(),
                                dtype: self.outputs[idx].dtype,
                            },
                        )
                        .unwrap();
                }
                _ => {
                    panic!("Unable to obtain output!");
                }

            }

            // return self.outputs[idx].get_ndarray(py);
        }
        // Err(())
        panic!("Error get_output!");

    }

    pub fn get_input_ndarray<'py>(&mut self, idx: usize, py: Python<'py>) -> &'py PyAny {
        info!("Operator::get_input idx {idx}");
        if idx < self.num_of_inputs() {
            // return Ok(&mut self.inputs[idx]);
            return self.inputs[idx].get_ndarray(py);
        }
        panic!("Error get_input!");
        // Err(())
    }

    pub fn get_output_ndarray<'py>(&mut self, idx: usize, py: Python<'py>) -> &'py PyAny{
        info!("Operator::get_output idx {idx}");
        if idx < self.num_of_outputs() {
            // return Ok(&self.outputs[idx]);
            return self.outputs[idx].get_ndarray(py);
        }
        // Err(())
        panic!("Error get_output!");

    }

    pub fn add_input(&mut self, x: TensorU, name: String, dtype: DataType) {
        self.inputs.push(Tensor {
            tensor: Some(x),
            name,
            dtype: dtype,
        });
    }

    pub fn get_unique_name(&self) -> String {
        let mut name = self.params["name"].clone();
        let idx = IDX.fetch_add(1, Ordering::SeqCst);
        name += idx.to_string().as_str();
        return name;
    }
    pub fn add_output(&mut self, x: TensorU, dtype: Option<DataType>) {
        
        let _dtype = match dtype {
            Some(t) => { t }
            _ => { DataType::Float }
        };

        // if !self.outputs.is_empty() {
            // let mut name = self.params["name"].clone();
            // let idx = IDX.fetch_add(1, Ordering::SeqCst);
            // name += idx.to_string().as_str();

            self.outputs.push(Tensor {
                tensor: Some(x),
                name: self.get_unique_name(),
                dtype: _dtype,
            });
        // } else {
        //     self.outputs.push(Tensor {
        //         tensor: Some(x),
        //         name: self.params["name"].to_string(),
        //         dtype: _dtype,
        //     });
        // }
    }

    pub fn calculate_output(&mut self) {
        info!("Operator::calculate_output for {:?}", self.op_type);
        if self.op_type!=OpType::ARANGE { //arange has no inputs
            assert!(!self.inputs.is_empty());
        } 
        match self.op_type {
            OpType::CONV2D => {
                let mut padding_w = 0;
                let mut padding_h = 0;
                let mut dilation_w = 1;
                let mut dilation_h = 1;
                if self.params.contains_key("pad") {
                    let padding : Vec<usize> = self.params["pad"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                    if padding.len() > 3 {
                        padding_h = padding[1];
                        padding_w = padding[3];
                    } else {
                        padding_h = padding[0];
                        padding_w = padding[1];
                    }
                }

                if self.params.contains_key("dilation") {
                    let dilation : Vec<usize> = self.params["dilation"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                    dilation_h = dilation[0];
                    dilation_w = dilation[1];
                }

                if self.params.contains_key("kernel")
                    && self.params.contains_key("stride")
                {
                    let kernel : Vec<usize> = self.params["kernel"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                    let kernel_h = kernel[0];
                    let kernel_w  = kernel[1];
                    let stride : Vec<usize> = self.params["stride"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                    let stride_h = stride[0];
                    let stride_w  = stride[1];

                    match &self.inputs[0].tensor {
                        Some(v) => {
                            info!("Input tensor shape {:?}", v.shape);
                            // let w = (v.shape[2] - kernel_w + 2 * padding_w) / stride_w + 1;
                            // let h = (v.shape[3] - kernel_h + 2 * padding_h) / stride_h + 1;
                            //formula [(W + 2*P - D*(K-1) - 1)/S]+1 , conv2d 2d
                            let h = (v.shape[2] + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
                            let w = (v.shape[3] + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
                            
                            let output_channel = if self.params.contains_key("out_channels") {
                                self.params["out_channels"].trim().parse::<usize>().unwrap()
                            } else {
                                if self.channel_first {v.shape[1]} else {v.shape[3]}
                            };

                            let output_shape = if self.channel_first {vec![v.shape[0], output_channel, h, w]} else {vec![v.shape[0], h, w, output_channel]};
                            let sz: usize = output_shape.iter().product();
                            info!("Output tensor with shape {:?} created within Rust for operator {:?}!", output_shape, self.op_type);
                            let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape);
                            self.add_output(tensor, Some(self.inputs[0].dtype));
                        }
                        _ => {
                            panic!("Invalid inputs!");
                        }
                    }
                } else {panic!("Missing important parameters!");}
            }
            OpType::MATMUL => {
                match (&self.inputs[0].tensor, &self.inputs[1].tensor) {
                    (Some(v1), Some(v2)) => {
                        let v1shape = v1.shape.clone();
                        let v2shape = v2.shape.clone();

                        if v1shape.len() != 2 || v2shape.len() != 2 || v1shape[1] != v2shape[0] {
                            panic!("Invalid shape a={:?}, b={:?}!", v1shape, v2shape);
                        }

                        let output_shape = vec![v1shape[0], v2shape[1]];
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape, self.op_type
                        );
                        let tensor = Operator::empty_tensor(v1.data_buffer.get_type_id(), output_shape.iter().product(), output_shape);

                        self.add_output(tensor, Some(self.inputs[0].dtype));
                    }
                    _ => {
                        panic!("Invalid inputs!");
                    }
                }
            }
            OpType::POOL2D => {
                if self.params.contains_key("pool_type") && 
                    (self.params["pool_type"].trim() == "PoolType.POOL_ADAPTIVE" ||
                    self.params["pool_type"].trim() == "PoolType.POOL_ADAPTIVE_AVG" ||
                    self.params["pool_type"].trim() == "PoolType.POOL_ADAPTIVE_MAX")  {
                    //formula [w,h]=output_size
                    match &self.inputs[0].tensor {
                        Some(v) => {
                            assert!(self.params.contains_key("output_size"));

                            let output_size : Vec<usize> = self.params["output_size"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                            let mut output_shape = v.shape.clone();
                            
                            if v.shape.len() <= 3 {
                                output_shape = if self.channel_first {vec![v.shape[0], v.shape[1], output_size[0]]} else {vec![v.shape[0], output_size[0], v.shape[3]]};
                            } else {
                                output_shape = if self.channel_first {vec![v.shape[0], v.shape[1], output_size[0], output_size[1]]} else {vec![v.shape[0], output_size[0], output_size[1], v.shape[3]]};
                            }
                            info!(
                                "Output tensor with shape {:?} created within Rust for operator {:?}!",
                                output_shape, self.op_type
                            );
                            let sz: usize = output_shape.iter().product();
                            let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape);
                            self.add_output(tensor, Some(self.inputs[0].dtype));
                        }
                        _ => {
                            panic!("Invalid inputs!");
                        }
                    }
                }
                else {
                    let mut padding_w = 0;
                    let mut padding_h = 0;
                    let mut dilation_w = 1;
                    let mut dilation_h = 1;

                    if self.params.contains_key("pad") {
                        // padding_w = self.params["padding_w"].trim().parse::<usize>().unwrap();
                        let padding : Vec<usize> = self.params["pad"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        padding_h = padding[0];
                        padding_w = padding[1];
                    }

                    if self.params.contains_key("dilation") {
                        // padding_w = self.params["padding_w"].trim().parse::<usize>().unwrap();
                        let dilation : Vec<usize> = self.params["dilation"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        dilation_h = dilation[0];
                        dilation_w = dilation[1];
                    }
    
                    if self.params.contains_key("kernel")
                        && self.params.contains_key("stride")
                    {
                        let kernel : Vec<usize> = self.params["kernel"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        let kernel_h = kernel[0];
                        let kernel_w  = kernel[1];
                        let stride : Vec<usize> = self.params["stride"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        let stride_h = stride[0];
                        let stride_w  = stride[1];
                        
                        
                        match &self.inputs[0].tensor {
                            Some(v) => {
                                //formula [(W + 2*P - D*(K-1) - 1)/S]+1 , max_pool 2d
                                let mut h = (v.shape[2] + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
                                let mut w = (v.shape[3] + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

                                if self.params.contains_key("pool_type") {
                                    if self.params["pool_type"].trim() == "PoolType.POOL_AVG" {
                                        //formula [(W + 2*P - K)/S]+1 , avg_pool 2d
                                        h = (v.shape[2] + 2 * padding_h - kernel_h) / stride_h + 1;
                                        w = (v.shape[3] + 2 * padding_w - kernel_w) / stride_w + 1;
                                    }
                                }

                                let output_shape = if self.channel_first {vec![v.shape[0], v.shape[1], h, w]} else {vec![v.shape[0], h, w, v.shape[3]]};
                                info!(
                                    "Output tensor with shape {:?} created within Rust for operator {:?}!",
                                    output_shape, self.op_type
                                );
                                let sz: usize = output_shape.iter().product();
                                let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape);
                                self.add_output(tensor, Some(self.inputs[0].dtype));
                            }
                            _ => {
                                panic!("Invalid inputs!");
                            }
                        }
                    }

                }
            }
            OpType::ARANGE => {

                let mut start : usize = 0;
                let mut step : usize = 1;
                let mut end : usize = 0;
                if self.params.contains_key("end") {
                    end = self.params["end"].trim().parse::<usize>().unwrap();
                } else {
                    panic!("Missing parameter 'end' for arange!");
                }

                if self.params.contains_key("start") {
                    start = self.params["start"].trim().parse::<usize>().unwrap();
                } 

                if self.params.contains_key("step") {
                    step = self.params["step"].trim().parse::<usize>().unwrap();
                } 

                let length = (end - start) / step;
                let output_shape = vec![1, length];

                let sz: usize = output_shape.iter().product();
                let tensor = Operator::empty_tensor(TypeId::of::<i32>(), sz, output_shape.clone());
                self.add_output(tensor, Some(DataType::Int32));
                info!(
                    "Output tensor with shape {:?} created within Rust for operator {:?}!",
                    output_shape,
                    self.op_type
                );
            }
            OpType::EMBEDDING => {

                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let mut embedding_dim : usize = 4096;
                        if self.params.contains_key("embedding_dim") {
                            embedding_dim = self.params["embedding_dim"].trim().parse::<usize>().unwrap();
                        } else {
                            panic!("Missing parameter 'embedding_dim' for embedding!");
                        }
    
                        let output_shape = vec![v.shape[0], v.shape[1], embedding_dim];
                        
                        let sz: usize = output_shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());

                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for embedding!");
                    }
                }

            }
            OpType::EXP
            | OpType::ADD
            | OpType::SUBTRACT
            | OpType::DIVIDE
            | OpType::DROPOUT
            | OpType::ELU
            | OpType::RELU
            | OpType::GELU
            | OpType::SILU
            | OpType::HARDSWISH
            | OpType::HARDSIGMOID
            | OpType::SIGMOID
            | OpType::TANH
            | OpType::IDENTITY
            | OpType::SCALAR_ADD
            | OpType::SCALAR_FLOORDIV
            | OpType::SCALAR_MULTIPLY
            | OpType::SCALAR_SUB
            | OpType::SCALAR_TRUEDIV
            | OpType::SOFTMAX
            | OpType::BATCH_NORM
            | OpType::LAYER_NORM
            | OpType::UNIFORM_LIKE
            | OpType::LESS 
            | OpType::CAST //TODO results cast into given type
            | OpType::EQ //output shape of equal op (compare a tensor with a scalar) is the shape of input
            | OpType::MASKEDFILL //output shape of masked_fill is equal to the input
            | OpType::MULTIHEAD_ATTENTION //output shape of multiheaded attention is equal to shape of input "q" (first item in input list) when batch_first=True
            | OpType::CLIP
            | OpType::ERF
            | OpType::BOOL
            | OpType::INVERT
            | OpType::AND
            | OpType::FLOAT
            | OpType::DETACH
            | OpType::CUMSUM
            | OpType::SQRT
            | OpType::RSQRT
            | OpType::RECIPROCAL
            | OpType::NEG
             => {
                // let activations = vec![OpType::ELU, OpType::RELU, OpType::GELU, OpType::SIGMOID, OpType::TANH];
                // let inplace = if self.params.contains_key("inplace") {self.params["inplace"]=="True"} else {false};
                // if false && activations.iter().any(|&act| act==self.op_type)  {
                //        //TODO: implements inplace operator for activations
                // } else {
                match &self.inputs[0].tensor {
                    Some(v) => {
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            v.shape, self.op_type
                        );
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), v.shape.iter().product(), v.shape.clone());
                        
                        if OpType::LESS == self.op_type {
                            self.add_output(tensor, Some(DataType::Bool));
                        } else {
                            self.add_output(tensor, Some(self.inputs[0].dtype));
                        }
                    }
                    _ => {
                        panic!("Invalid inputs!");
                    }
                }
                // }
            }
            OpType::MULTIPLY => {
                match (&self.inputs[0].tensor, &self.inputs[1].tensor) {
                    (Some(v1), Some(v2)) => {
                        let mut v1shape = v1.shape.clone();
                        let mut v2shape = v2.shape.clone();

                        if v1shape.len() != v2shape.len() {
    
                            let mut vshape = if v1shape.len() > v2shape.len() {v1shape.clone()} else {v2shape.clone()};
                            vshape = vshape[0..v1shape.len().abs_diff(v2shape.len())].to_vec();
                            vshape.extend(if v1shape.len() > v2shape.len() {v2shape.clone()} else {v1shape.clone()});
                            // info!("vshape {:?}", vshape);

                            if v1shape.len() > v2shape.len() {
                                v2shape = vshape;
                            }
                            else {
                                v1shape = vshape;
                            }
                            // panic!("Mismatched shape a={:?}, b={:?}!", v1.shape, v2.shape);
                        }
                        let mut output_shape = vec![];
                        for i in 0..v1shape.len() {
                            if v1shape[i] != v2shape[i] && v1shape[i] != 1 && v2shape[i] != 1 {
                                panic!("Mismatched shape a={:?}, b={:?} in dim {}!", v1shape, v2shape, i);
                            }
                            output_shape.push(if v1shape[i] > v2shape[i] {v1shape[i]} else {v2shape[i]}); 
                        }

                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape, self.op_type
                        );
                        let tensor = Operator::empty_tensor(v1.data_buffer.get_type_id(), output_shape.iter().product(), output_shape);
                        self.add_output(tensor, Some(self.inputs[0].dtype));
                    }
                    _ => {
                        panic!("Invalid inputs!");
                    }
                }
            }
            OpType::CONCAT => {
                //formula
                // w1 + w2 for axis 1
                // h1 + h2 for axis 0
                // let (mut batch, mut channel, mut w, mut h, mut batch_out, mut channel_out) = (0, 0, 0, 0, 0, 0);
                // let mut out_dim = 0;
                let mut output_shape = vec![0, 0, 0, 0];
                let idx_i = self.params["axis"].trim().parse::<i32>().unwrap();
                let mut idx : usize = 0;
                let mut typeid = TypeId::of::<f32>();
                if let Some(v) = &mut self.inputs[0].tensor {
                    output_shape = v.shape.clone();
                    typeid = v.data_buffer.get_type_id();

                    if idx_i < 0 {
                        idx = v.shape.len() - abs(idx_i) as usize;
                    } else {
                        idx = idx_i as usize;
                    }
                }

                output_shape[idx] = 0;

                for tensor in &self.inputs {
                    if let Some(v) = &tensor.tensor {
                        output_shape[idx] += v.shape[idx];
                    }
                }

                info!(
                    "Output tensor with shape {:?} created within Rust for operator {:?}!",
                    output_shape, self.op_type
                );
                let sz: usize = output_shape.iter().product();
                let tensor = Operator::empty_tensor(typeid, sz, output_shape);
                self.add_output(tensor, Some(self.inputs[0].dtype));
            }
            OpType::SPLIT | OpType::CHUNK => {
                //formula 1 for sizes of int
                //sizes of C/sizes for axis 1 (channel_first)

                //formula 2 for sizes of list (channel_first)
                //C1 -> list.1
                //C2 -> list.2
                //..
                let mut typeid = TypeId::of::<f32>();
                match &self.inputs[0].tensor {
                    Some(v) => {
                        typeid = v.data_buffer.get_type_id();
                    }
                    _=>{}
                }
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let params = self.params.clone();
                        let mut idx = params["axis"].trim().parse::<i32>().unwrap();
                        if idx == -1 {
                            idx = v.shape.len() as i32 - 1;
                        }
                        let mut output_shape = v.shape.clone();
                        if params.contains_key("sizes") {
                            let mut sizes = vec![];
                            if params["sizes"].find("[").is_some() { //sizes of list
                                let parts = params["sizes"].replace(['[',']'], "");                                
                                for part in parts.split([',']) {
                                    let size = part.trim().parse::<usize>().unwrap();
                                    sizes.push(size);
                                }
                            }
                            else {
                                let mut size = params["sizes"].trim().parse::<usize>().unwrap();
                                let mut eachsize = size;
                                if self.op_type == OpType::CHUNK {
                                    eachsize = output_shape[idx as usize] / size;
                                    for _ in 0..size {
                                        sizes.push(eachsize);
                                    }
                                } else {
                                    size = output_shape[idx as usize] / eachsize;
                                    for _ in 0..size {
                                        sizes.push(eachsize);
                                    }
                                }
                                    
                                if sizes.len() * eachsize < output_shape[idx as usize] { 
                                    if self.op_type == OpType::CHUNK { //the remaining parts for chunk
                                        sizes.push(output_shape[idx as usize] - sizes.len() * eachsize);
                                    }
                                    else {
                                        panic!("Unable to split on the given axis!");
                                    }
                                }
                            }

                            for size in sizes {
                                output_shape[idx as usize] = size;
                                let sz: usize = output_shape.iter().product();
                                let tensor = Operator::empty_tensor(typeid, sz, output_shape.clone());
                                self.add_output(tensor, Some(self.inputs[0].dtype));
                                info!("Output tensor with shape {:?} created within Rust for operator {:?}!", output_shape, self.op_type);
                            }

                        } else {
                            panic!("Missing parameter 'sizes' for split/chunk!");
                        };
                    }
                    _ => {
                        panic!("Invalid tensor for axis {} split/chunk!", self.params["axis"]);
                    }
                }
            }
            OpType::FLAT => {
                //formula
                // output shape = [batch, channel * w * h]
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let mut start_dim = 0;
                        let mut end_dim = v.shape.len() - 1;
                        if self.params.contains_key("start_dim") {
                            start_dim = self.params["start_dim"].trim().parse::<usize>().unwrap();
                        }
                        if self.params.contains_key("end_dim") {
                            let dim = self.params["end_dim"].trim().parse::<i32>().unwrap();
                            if dim > -1 {
                                end_dim = dim as usize;
                            } 
                        }
                        let mut output_shape = vec![];
                        let mut reduced_dim = 1;
                        for i in 0..v.shape.len() {
                            if i >= start_dim && i <= end_dim {
                                reduced_dim *= v.shape[i];
                                if i == end_dim {
                                    output_shape.push(reduced_dim);
                                }
                            } 
                            else {
                                output_shape.push(v.shape[i]);
                            }
                        }
                        let sz: usize = v.shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());
                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for flat!");
                    }
                }
            }
            OpType::RESHAPE => {
                //formula
                // output shape = [batch, channel * w * h]
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        assert!(self.params.contains_key("shape"));

                        let output_shape : Vec<usize> = self.params["shape"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        let sz: usize = v.shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());
                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for flat!");
                    }
                }
            }

            OpType::TRANSPOSE => {
                //formula
                // output shape = input_shape[perm2, perm1, perm0, ...]
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        assert!(self.params.contains_key("perms"));

                        let output_idx : Vec<usize> = self.params["perms"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        let mut output_shape : Vec<usize>  = vec![];
                        for i in 0..v.shape.len() {
                            let idx = output_idx[i];
                            output_shape.extend([v.shape.get(idx).unwrap()]);
                        }
                        let sz: usize = v.shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());
                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for flat!");
                    }
                }
            }

            OpType::EXPAND => {
                //formula
                //input_shape = [1, 3, 224]
                //sizes = [5, -1, -1]
                //outpus_shape = [4, 3, 224]

                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        assert!(self.params.contains_key("sizes"));

                        let output_idx : Vec<i32> = self.params["sizes"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<i32>().unwrap()).collect();
                        let mut output_shape : Vec<usize>  = v.shape.clone();
                        for i in 0..v.shape.len() {
                            let dim = output_idx[i];
                            output_shape[i]= if dim!=-1 {dim as usize} else {v.shape[i]};
                        }
                        let sz: usize = v.shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());
                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for expand!");
                    }
                }
            }

            OpType::LINEAR => {
                //formula
                // output shape = [batch, .. output dim]
                if !self.params.contains_key("out_dim") {
                    panic!("Missing important parameters!");
                }

                let output_dim = self.params["out_dim"].trim().parse::<usize>().unwrap();
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let mut output_shape = v.shape.clone(); //vec![v.shape[0], output_dim, v.shape[2], v.shape[3]];
                        output_shape[v.shape.len() - 1] = output_dim;
                        let sz: usize = output_shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());
                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for flat!");
                    }
                }
            }
            OpType::SLICE => {
                //formula
                //output_shape = input_shape[slices]
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        assert!(self.params.contains_key("output_shape"));

                        let output_shape : Vec<usize> = self.params["output_shape"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();

                        let sz: usize = output_shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());

                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for flat!");
                    }
                }
            }

            OpType::PARAMETER | OpType::TENSOR => {
                //formula
                //output_shape = input_shape
                self.outputs.append(&mut self.inputs);
                self.outputs[0].name = self.get_unique_name();

                // match &mut self.inputs[0].tensor {
                //     Some(v) => {
                //         let output_shape = v.shape.clone();
                //         let sz: usize = output_shape.iter().product();

                //         let tensor = TensorU {
                //             shape: output_shape.clone(),
                //             data_buffer: v.data_buffer.clone()
                //         };
                //         self.add_output(tensor, Some(self.inputs[0].dtype));
                //         info!(
                //             "Output tensor with shape {:?} created within Rust for operator {:?}!",
                //             output_shape,
                //             self.op_type
                //         );
                //     }
                //     _ => {
                //         panic!("Invalid tensor for parameter!");
                //     }
                // }
            }
            OpType::MEAN => {
                //keepdims
                if !self.params.contains_key("dims") && !self.params.contains_key("keepdims") {
                    panic!("Missing important parameters!");
                }
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let mut keepdims = false;
                        if self.params.contains_key("keepdims") {
                            if self.params["keepdims"].trim() == "True" {
                                keepdims = true;
                            }
                        } 

                        let mut output_shape: Vec<usize> = vec![];
                        if self.params.contains_key("dims") {
                            if self.params["dims"].find("[").is_some() {
                                let dims : Vec<usize> = self.params["dims"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                                for dim in 0..v.shape.len() {
                                    if !dims.contains(&dim) {
                                        output_shape.push(v.shape[dim]);
                                    } else if keepdims {
                                        output_shape.push(1);
                                    }
                                }
                            } else {
                                let dims = self.params["dims"].trim().parse::<usize>().unwrap();
                                for dim in 0..v.shape.len() {
                                    if dim != dims {
                                        output_shape.push(v.shape[dim]);
                                    } else if keepdims {
                                        output_shape.push(1);
                                    }
                                }
                            }
                        } 

                        let sz: usize = output_shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());
                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for mean!");
                    }
                }
            }

            OpType::BATCH_MATMUL => {
                match (&self.inputs[0].tensor, &self.inputs[1].tensor) {
                    (Some(v1), Some(v2)) => {
                        let v1shape = v1.shape.clone();
                        let v2shape = v2.shape.clone();

                        if v1shape.len() < 3 || v2shape.len() < 3 {
                            panic!("Invalid shape a={:?}, b={:?}!", v1shape, v2shape);
                        }

                        let m1shape = &v1shape[v1shape.len()-2..];
                        let m2shape = &v2shape[v2shape.len()-2..];
                        if m1shape[1] != m2shape[0] {
                            panic!("Mismatched shape a={:?}, b={:?}!", v1shape, v2shape);
                        }

                        let mut idx : usize = 0;

                        for i in 0..if v1shape.len() > v2shape.len() {v2shape.len()} else {v1shape.len()} {
                            if v1shape[i] == v2shape[i] {
                                idx += 1;
                            }
                        }

                        let mut output_shape : Vec<usize> = v1shape[0..v1shape.len()-2].to_vec();
                        output_shape.push(m1shape[0]);

                        if idx  < v2shape.len() - 2 {
                            output_shape.extend(v2shape[idx..v2shape.len() - 2].iter());
                        }

                        output_shape.push(m2shape[1]);

                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape, self.op_type
                        );
                        let sz = output_shape.iter().product();
                        let tensor = Operator::empty_tensor(v1.data_buffer.get_type_id(), sz, output_shape);

                        self.add_output(tensor, Some(self.inputs[0].dtype));
                    }
                    _ => {
                        panic!("Invalid inputs!");
                    }
                }
            }

            OpType::REPEAT => {
                //keepdims
                if !self.params.contains_key("sizes") {
                    panic!("Missing important parameters!");
                }
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let mut output_shape: Vec<usize> = vec![];

                        let repeats : Vec<usize> = self.params["sizes"].replace(['[', ']'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        if repeats.len() != v.shape.len() {
                            panic!("Invalid sizes {:?} for repeat!", repeats);
                        }
                        for i in 0..repeats.len() {
                            output_shape.push(v.shape[i] * repeats[i]);
                        }
               

                        let sz: usize = output_shape.iter().product();
                        let tensor = Operator::empty_tensor(v.data_buffer.get_type_id(), sz, output_shape.clone());
                        self.add_output(tensor, Some(self.inputs[0].dtype));
                        info!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            output_shape,
                            self.op_type
                        );
                    }
                    _ => {
                        panic!("Invalid tensor for repeat!");
                    }
                }
            }

            _ => {
                panic!("Not implemented!");
            }
        }
    }

    pub fn memory_usage(&self) -> usize {
        let mut memory_size: usize = 0;
        for input in &self.inputs {
            if let Some(tensor) = &input.tensor {
                let sz: usize = tensor.shape.iter().product();
                memory_size += sz;
            }
        }
        for output in &self.outputs {
            if let Some(tensor) = &output.tensor {
                let sz: usize = tensor.shape.iter().product();
                memory_size += sz;
            }
        }

        memory_size * 4 //float32
    }

    pub fn get_ir(&self) -> String {
        return self.dump_ir(None);
    }

    fn parse_call_args(&self, s: String) -> Vec<String> {
        let mut strargs = s.replace("OrderedDict([", "").to_string(); //.replace("])", "");
        strargs = strargs[0..strargs.len() - 2].to_string();

        let lst: Vec<&str> = strargs.split("), ").collect();
        let mut retlst: Vec<String> = Vec::<String>::new();
        for item in lst {
            match item.find("Tensor") {
                Some(_) => {
                    continue;
                }
                _ => {
                    retlst.push(item.to_string());
                }
            }
        }

        for i in 0..retlst.len() {
            let item = retlst[i].to_string();
            if item[item.len() - 1..] != ")".to_string() {
                retlst[i] += ")";
            }
            if i < retlst.len() - 1 {
                retlst[i] += ", ";
            }
        }
        retlst
    }
    pub fn dump_ir(&self, ssa_ids: Option<&HashMap<String, i32>>) -> String {
        let name = self.op_type.as_str();
        let mut params = self.params.clone();
        params.remove("name");
        params.remove("tensors"); //for concat
        params.remove("out_dim");
        params.remove("out_channels");
        params.remove("x"); //for add
        params.remove("y"); //for add

        if self.op_type == OpType::MULTIHEAD_ATTENTION {
            params.remove("q");
            params.remove("k");
            params.remove("v");
            if params.contains_key("batch_first") {
                params["batch_first"] = params["batch_first"].to_lowercase();
            }
        } else if self.op_type == OpType::PARAMETER || self.op_type == OpType::TENSOR {
            // params.remove("dtype");
            params.remove("np_tensor");
            if params.contains_key("requires_grad") {
                params["requires_grad"] = params["requires_grad"].to_lowercase();
            }
            if params.contains_key("initializer") {
                params["initializer"] = params["initializer"].replace('\n', "");
            } else {
                match &self.outputs[0].tensor {
                    Some(v) => match &v.data_buffer { 
                        DataBuffer::CPUDataBuffer(data) => {
                            if self.outputs[0].dtype == DataType::Float {
                                params.insert("initializer".to_string(), format!("{:?}", data.as_ptr::<f32>()));
                            } else {
                                params.insert("initializer".to_string(), format!("{:?}", data.as_ptr::<f16>()));
                            }
                            // let v = data.to_vec()[100];
                            // let memaddr = format!("{:?}", data.as_ptr());
                            // println!("Rust address {memaddr} 100th value {v}");
                        }
                        _ => panic!("Unable to init parameter, tensor conversion failed!"),
                    },
                    _ => panic!("Unable to init parameter, tensor not initialized!"),
                }
            }
        } else if self.op_type == OpType::CALL {
            params.remove("argtypes");
            params.remove("callback");
            if params.contains_key("args") {
                let lst = self.parse_call_args(params["args"].to_string());
                params.remove("args");
                let mut s = "[".to_string();
                for item in lst {
                    s += &item;
                }
                s += "]";
                params.insert("addargs".to_string(), s);
            }
        } else if self.op_type == OpType::MASKEDFILL {
            params.remove("mask");
        } else if self.op_type == OpType::LAYER_NORM {
            //elementwise_affine True/False -> true/false
            if params.contains_key("elementwise_affine") {
                params["elementwise_affine"] = params["elementwise_affine"].to_lowercase();
            }
            //normalized_shape
            //eps
            if params.contains_key("eps") {
                let eps = params["eps"].parse::<f32>().unwrap();
                params["eps"] = eps.to_string();
            }
        } else if self.op_type == OpType::BATCH_NORM {
            //track_running_stats True/False -> true/false
            if params.contains_key("track_running_stats") {
                params["track_running_stats"] = params["track_running_stats"].to_lowercase();
            }
            //affine True/False -> true/false
            if params.contains_key("affine") {
                params["affine"] = params["affine"].to_lowercase();
            }
            //eps
            if params.contains_key("eps") {
                let eps = params["eps"].parse::<f32>().unwrap();
                params["eps"] = eps.to_string();
            }
        } else if self.op_type == OpType::MEAN {
            if params.contains_key("keepdims") {
                params["keepdims"] = params["keepdims"].to_lowercase();
            }
        } else if self.op_type == OpType::GELU || self.op_type == OpType::ERF {
            if params.contains_key("approximate") {
                params["approximate"] = params["approximate"].to_lowercase();
            }
        } else if self.op_type == OpType::DROPOUT {
            if params.contains_key("training") {
                params["training"] = params["training"].to_lowercase();
            }
        }

        if self.op_type == OpType::CONV2D || self.op_type == OpType::LINEAR{
            params.remove("weight");
            params.remove("bias");
            let mut segment_sizes = vec![0; 3];
            for i in 0..self.inputs.len() {
                segment_sizes[i] = 1;
            }
            let mut strsize = format!("{:?}", segment_sizes);
            strsize = strsize.replace("[", "").replace("]", "");
            params.insert("operand_segment_sizes".to_string(), format!("array<i32:{strsize}>"));
        }

        if self.op_type == OpType::MULTIHEAD_ATTENTION{
            params.remove("weight_q");
            params.remove("weight_k");
            params.remove("weight_v");
            params.remove("bias_q");
            params.remove("bias_k");
            params.remove("bias_v");
            params.remove("weight_o");
            params.remove("bias_o");
            
            let mut segment_sizes = vec![0; 12];
            for i in 0..self.inputs.len() {
                segment_sizes[i] = 1;
            }
            let mut strsize = format!("{:?}", segment_sizes);
            strsize = strsize.replace("[", "").replace("]", "");
            params.insert("operand_segment_sizes".to_string(), format!("array<i32:{strsize}>"));
            // println!("array<i64:{strsize}>")
        }

        if self.op_type == OpType::BATCH_NORM || self.op_type == OpType::LAYER_NORM{
            params.remove("weight");
            params.remove("bias");
            params.remove("mean");
            params.remove("variance");
            
            let mut segment_sizes = if self.op_type == OpType::BATCH_NORM {vec![0; 5]} else {vec![0; 3]};
            for i in 0..self.inputs.len() {
                segment_sizes[i] = 1;
            }
            let mut strsize = format!("{:?}", segment_sizes);
            strsize = strsize.replace("[", "").replace("]", "");
            params.insert("operand_segment_sizes".to_string(), format!("array<i32:{strsize}>"));
        }

        params.remove("activation"); //fused activation not supported at the moment
        params.remove("inplace"); //inplace not supported at the moment
        params.remove("use_bias"); //bias supported at the moment

        let mut input_names = "".to_string();
        let mut input_shapes = "".to_string();

        for input in &self.inputs {
            input_names += "%";
            match ssa_ids {
                Some(ssa) => {
                    if ssa.contains_key(&input.name) {
                        input_names += &ssa[&input.name].to_string();
                    } else {
                        input_names += input.name.as_str();
                    }
                }
                _ => {
                    input_names += input.name.as_str();
                }
            }

            input_names += ", ";

            input_shapes += input.get_ir().as_str();
            input_shapes += ", ";
        }

        if self.inputs.len() > 0 {
            input_names = input_names.trim().to_string();
            if &input_names[input_names.len() - 1..] == "," {
                input_names.pop();
            }
    
            input_shapes = input_shapes.trim().to_string();
            if &input_shapes[input_shapes.len() - 1..] == "," {
                input_shapes.pop();
            }
        }

        let mut output_names = "".to_string();
        let mut output_shapes = "".to_string();

        if self.outputs.len() > 1 {
            output_shapes += "(";
        }
        for output in &self.outputs {
            output_names += "%";
            match ssa_ids {
                Some(ssa) => {
                    if ssa.contains_key(&output.name) {
                        output_names += &ssa[&output.name].to_string();
                    } else {
                        output_names += output.name.as_str();
                    }
                }
                _ => {
                    output_names += output.name.as_str();
                }
            }

            output_names += ", ";

            output_shapes += output.get_ir().as_str();
            output_shapes += ", ";
        }

        if self.outputs.len() > 0 {
            output_names = output_names.trim().to_string();
            if &output_names[output_names.len() - 1..] == "," {
                output_names.pop();
            }
    
            output_shapes = output_shapes.trim().to_string();
            if &output_shapes[output_shapes.len() - 1..] == "," {
                output_shapes.pop();
            }
        }

        if self.outputs.len() > 1 {
            output_shapes += ")";
        }

        if self.op_type == OpType::PARAMETER || self.op_type == OpType::TENSOR { //no inputs for these two ops
            input_names = "".to_string();
            input_shapes = "".to_string();
        }

        if params.is_empty() {
            let ir = format!("{output_names}=\"ufront.{name}\"({input_names}):({input_shapes}) -> {output_shapes}");
            ir
        } else {
            let mut params_str = "{".to_string();
            for (key, v) in params.iter().sorted() {
                if key == "pool_type" || key == "ActiMode" || key == "dtype" {
                    let mut s = v.clone();
                    s = s.replace("PoolType.", "");
                    s = s.replace("ActiMode.", "");
                    s = s.replace("DataType.", "");

                    params_str += format!("{key}=\"{s}\"").as_str();
                } else if key == "initializer" {
                    params_str += format!("{key}=\"{v}\"").as_str();
                }
                else {
                    params_str += format!("{key}={v}").as_str();
                };
                params_str += ", ";
            }
            params_str = params_str[0..params_str.len() - 2].to_string();
            params_str += "}";

            let ir = format!("{output_names}=\"ufront.{name}\"({input_names}){params_str}:({input_shapes}) -> {output_shapes}");
            ir
        }
    }
}

#[pyclass]
pub struct PyOperator {
    // #[pyo3(get, set)]
    // pub id : usize,
    #[pyo3(get, set)]
    pub op_type: OpType,
    #[pyo3(get, set)]
    pub params: HashMap<String, String>,
    #[pyo3(get)]
    pub raw_ptr: u64,
}

#[pymethods]
impl PyOperator {
    #[new]
    pub fn new(
        op_type: OpType,
        params: HashMap<String, String>,
    ) -> PyResult<PyClassInitializer<Self>> {
        info!("PyOperator::new");
        let op = PyOperator {
            op_type,
            params,
            raw_ptr: 0,
        };
        Ok(PyClassInitializer::from(op))
    }
    pub fn num_of_inputs(&self) -> usize {
        if self.raw_ptr > 0 {
            let operator = self.raw_ptr as *const Operator;
            unsafe { (*operator).num_of_inputs() }
        } else {
            0
        }
    }

    pub fn num_of_outputs(&self) -> usize {
        if self.raw_ptr > 0 {
            let operator = self.raw_ptr as *const Operator;
            unsafe { (*operator).num_of_outputs() }
        } else {
            0
        }
    }

    pub fn get_input_ndarray<'py>(&self, idx: usize, py: Python<'py>) -> &'py PyAny {
        info!("Operator::get_input_ndarray idx {idx}");
        if self.raw_ptr > 0 && idx < self.num_of_inputs() {
            let operator = self.raw_ptr as *mut Operator;
            unsafe {
                return (*operator).get_input_ndarray(idx, py);
                // match &mut (*operator).get_input(idx) {
                //     Ok(&mut tensor) => {
                //         tensor.get_ndarray(py)
                //     }
                    // Ok(tensor) => match &tensor.tensor {
                    //     Some(v) => match &v.data_buffer {
                    //         DataBuffer::CPUDataBuffer(data) => {
                    //             ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec())
                    //                 .unwrap()
                    //                 .into_pyarray(py)
                    //         }
                    //         _ => panic!("Tensor conversion failed!"),
                    //     },
                    //     _ => panic!("Tensor not initialized!"),
                    // },
                //     _ => panic!("Tensor not initialized!"),
                // }
            }
        } else {
            panic!("Tensor not initialized or index out of bound!")
        }
    }

    pub fn get_output_ndarray<'py>(&self, idx: usize, py: Python<'py>) -> &'py PyAny {
        info!("Operator::get_output_ndarray {idx}");
        if self.raw_ptr > 0 && idx < self.num_of_outputs() {
            let operator = self.raw_ptr as *mut Operator;
            unsafe {
                return (*operator).get_output_ndarray(idx, py);
                // match (*operator).get_output(idx) {
                //     Ok(tensor) => match &tensor.tensor {
                //         Some(v) => match &v.data_buffer {
                //             DataBuffer::CPUDataBuffer(data) => {
                //                 ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec())
                //                     .unwrap()
                //                     .into_pyarray(py)
                //             }
                //             _ => panic!("Tensor conversion failed!"),
                //         },
                //         _ => panic!("Tensor not initialized!"),
                //     },
                //     _ => panic!("Tensor not initialized!"),
                // }
            }
        } else {
            panic!("Tensor not initialized or index out of bound!")
        }
    }

    pub fn add_input_ndarray(&mut self, x: PyReadonlyArrayDyn<f32>, name: String) {
        let x = x.as_array();
        match x.as_slice() {
            Some(v) => {
                let tensor = TensorU {
                    shape: x.shape().to_vec(),
                    data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(v.len(), Some(v.to_vec()))),
                };

                if self.raw_ptr > 0 {
                    unsafe {
                        let operator = std::mem::transmute::<u64, *mut Operator>(self.raw_ptr);
                        (*operator).add_input(tensor, name, DataType::Float);
                    }
                } else {
                    info!("Input not added to the graph!");
                }

                info!(
                    "Tensor initialized with {:?} dimension within Rust",
                    x.shape().to_vec()
                );
            }
            _ => panic!("Invalid tensor inputs!"),
        }
    }

    pub fn add_input(&mut self, x: &Tensor) -> PyResult<()> {
        info!("Operator::add_input");
        if self.raw_ptr > 0 {
            unsafe {
                let operator = std::mem::transmute::<u64, *mut Operator>(self.raw_ptr);
                match &x.tensor {
                    Some(v) => {
                        let tensor = TensorU {
                            shape: v.shape.to_vec(),
                            data_buffer: v.data_buffer.clone(),
                        };
                        (*operator).add_input(tensor, x.name.clone(), x.dtype);
                        Ok(())
                    }
                    _ => {
                        panic!("Invalid tensor inputs!")
                    }
                }
            }
        } else {
            panic!("Input not added to the graph!");
        }
    }

    pub fn calculate_output(&mut self) {
        if self.raw_ptr > 0 {
            unsafe {
                let operator = std::mem::transmute::<u64, *mut Operator>(self.raw_ptr);
                (*operator).calculate_output();
            }
        } else {
            panic!("Input not added to the graph!");
        }
    }

    pub fn add_output(&mut self, x: &Tensor) {
        info!("Operator::add_output");
        if self.raw_ptr > 0 {
            unsafe {
                let operator = std::mem::transmute::<u64, *mut Operator>(self.raw_ptr);
                match &x.tensor {
                    Some(v) => {
                        let tensor = TensorU {
                            shape: v.shape.to_vec(),
                            data_buffer: v.data_buffer.clone(),
                        };
                        (*operator).add_output(tensor, Some(x.dtype));
                    }
                    _ => {
                        panic!("Invalid tensor outputs!")
                    }
                }
            }
        } else {
            panic!("OUtput not added to the graph!");
        }
    }
    pub fn get_input<'py>( &self, idx : usize, py: Python<'py>) -> Py<Tensor> {
        if self.raw_ptr > 0 {
            let operator = self.raw_ptr as *mut Operator;
            unsafe {
                return (*operator).get_input(idx, py);
            }
        }
        else {
            panic!("Invalid operator pointer!");
        }
    }

    pub fn get_output<'py>(&self, idx: usize, py: Python<'py>) -> Py<Tensor> {
        if self.raw_ptr > 0 {
            let operator = self.raw_ptr as *mut Operator;
            unsafe {
                return (*operator).get_output(idx, py);
                // match (*operator).get_output(idx) {
                //     Ok(tensor) => match &tensor.tensor {
                //         Some(v) => {
                //             let tensor_ = TensorU {
                //                 shape: v.shape.to_vec(),
                //                 data_buffer: v.data_buffer.clone(),
                //             };

                //             Python::with_gil(|py| {
                //                 Py::new(
                //                     py,
                //                     Tensor {
                //                         tensor: Some(tensor_),
                //                         name: tensor.name.clone(),
                //                         dtype: tensor.dtype,
                //                     },
                //                 )
                //                 .unwrap()
                //             })
                //         }
                //         _ => {
                //             panic!("Unable to obtain output!");
                //         }
                //     },

                //     _ => {
                //         panic!("Unable to obtain output!");
                //     }
                // }
            }
        } else {
            panic!("Invalid operator pointer!");
        }
    }

    pub fn memory_usage(&self) -> usize {
        if self.raw_ptr > 0 {
            let operator = self.raw_ptr as *mut Operator;
            unsafe { (*operator).memory_usage() }
        } else {
            panic!("Invalid operator pointer!");
        }
    }

    #[getter]
    pub fn get_ir(&self) -> PyResult<String> {
        if self.raw_ptr > 0 {
            let operator = self.raw_ptr as *mut Operator;
            unsafe { Ok((*operator).get_ir()) }
        } else {
            panic!("Invalid operator pointer!");
        }
    }
}
// #[pymethods]
// impl PyOperator {

// }
