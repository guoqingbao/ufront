use crate::databuffer::DataBuffer;
use crate::tensor::Tensor;
use crate::tensor::TensorF32;
use crate::types::DataType;
use crate::types::OpType;
use core::panic;
use ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
// use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
// use std::os::raw::c_void;
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
    pub inputs: Vec<TensorF32>,
    pub outputs: Vec<TensorF32>,
    pub op_type: OpType,
    pub params: HashMap<String, String>,
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
    pub fn new(op: OpType, params: HashMap<String, String>) -> Operator {
        println!("Operator::new------------------------------------------------------------------");
        // let mut params_ : HashMap<String, String> = HashMap::new();
        // params_.extend(params.into_iter());
        // Python::with_gil(|py| {
        //     let inputs: Py<TensorF32> = Py::new(py, TensorF32 { tensor : None})?;
        //     let outputs: Py<TensorF32> = Py::new(py, TensorF32 { tensor : None})?;

        //     let operator = Operator {id : id, inputs : inputs, outputs : outputs, op_type : op, params : params_};
        //     // return operator;
        //     Ok(PyClassInitializer::from(operator))
        // })
        Operator {
            inputs: Vec::new(),
            outputs: Vec::new(),
            op_type: op,
            params,
        }
    }

    pub fn num_of_inputs(&self) -> usize {
        //
        let num = self.inputs.len();
        println!("Operator::num_of_inputs {num}");
        num
    }

    pub fn num_of_outputs(&self) -> usize {
        let num = self.outputs.len();
        println!("Operator::num_of_outputs {num}");
        num
    }

    // pub fn get_input_ndarray<'py>(&self, idx : usize, py: Python<'py>) -> &'py PyArrayDyn<f32>{
    //     println!("Operator::get_input_ndarray {}", idx);
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

    // pub fn get_input(&self, idx : usize) -> Result<&TensorF32, ()> {

    //     Err(())
    //     // return Err(PyOSError::new_err("Failed to obtain tensor!".to_string()))
    // }

    pub fn get_input(&self, idx: usize) -> Result<&TensorF32, ()> {
        println!("Operator::get_input idx {idx}");
        if idx < self.num_of_inputs() {
            return Ok(&self.inputs[idx]);
        }
        Err(())

        // return Err(PyOSError::new_err("Failed to obtain tensor!".to_string()))
    }

    pub fn get_output(&self, idx: usize) -> Result<&TensorF32, ()> {
        println!("Operator::get_output idx {idx}");
        if idx < self.num_of_outputs() {
            return Ok(&self.outputs[idx]);
        }
        Err(())

        // return Err(PyOSError::new_err("Failed to obtain tensor!".to_string()))
    }

    pub fn add_input(&mut self, x: Tensor<f32>, name: String) {
        self.inputs.push(TensorF32 { tensor: Some(x), name });
    }

    pub fn add_output(&mut self, x: Tensor<f32>) {
        if !self.outputs.is_empty() {
            let name = format!("{}_{}", self.params["name"], self.outputs.len());
            self.outputs.push(TensorF32 { tensor: Some(x), name });
        }else {
            self.outputs.push(TensorF32 { tensor: Some(x), name: self.params["name"].to_string() });
        }
    }

    pub fn calculate_output(&mut self) {
        println!("Operator::calculate_output for {:?}", self.op_type);
        assert!(!self.inputs.is_empty());
        match self.op_type {
            OpType::CONV2D => {
                //formula [(W−K+2P)/S]+1
                let mut padding_w = 0;
                let mut padding_h = 0;
                if self.params.contains_key("padding_w") {
                    padding_w = self.params["padding_w"].trim().parse::<usize>().unwrap();
                }
                if self.params.contains_key("padding_h") {
                    padding_h = self.params["padding_h"].trim().parse::<usize>().unwrap();
                }

                if self.params.contains_key("kernel_h")
                    && self.params.contains_key("kernel_w")
                    && self.params.contains_key("stride_h")
                    && self.params.contains_key("stride_w")
                {
                    let kernel_h = self.params["kernel_h"].trim().parse::<usize>().unwrap();
                    let kernel_w = self.params["kernel_w"].trim().parse::<usize>().unwrap();
                    let stride_h = self.params["stride_h"].trim().parse::<usize>().unwrap();
                    let stride_w = self.params["stride_w"].trim().parse::<usize>().unwrap();
                    match &self.inputs[0].tensor {
                        Some(v) => {
                            println!("Input tensor shape {:?}", v.shape);
                            let w = (v.shape[2] - kernel_w + 2 * padding_w) / stride_w + 1;
                            let h = (v.shape[3] - kernel_h + 2 * padding_h) / stride_h + 1;
                            let output_channel = if self.params.contains_key("out_channels") {
                                self.params["out_channels"].trim().parse::<usize>().unwrap()
                            } else {
                                v.shape[1]
                            };

                            let output_shape = vec![v.shape[0], output_channel, w, h];
                            let sz: usize = output_shape.iter().product();
                            println!("Output tensor with shape {:?} created within Rust for operator {:?}!", output_shape, self.op_type);
                            let tensor = Tensor::<f32> {
                                shape: output_shape,
                                data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                            };
                            self.add_output(tensor);
                        }
                        _ => {
                            panic!("Invalid inputs!");
                        }
                    }
                } else {panic!("Missing important parameters!");}
            }
            OpType::POOL2D => {
                if self.params.contains_key("pool_type") && self.params["pool_type"].trim() == "PoolType.POOL_ADAPTIVE" {
                    //formula (w,h)=output_size
                    match &self.inputs[0].tensor {
                        Some(v) => {
                            assert!(self.params.contains_key("output_size"));

                            let output_size : Vec<usize> = self.params["output_size"].replace(['(', ')'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();

                            let output_shape = vec![v.shape[0], v.shape[1], output_size[0], output_size[1]];
                            println!(
                                "Output tensor with shape {:?} created within Rust for operator {:?}!",
                                output_shape, self.op_type
                            );
                            let sz: usize = output_shape.iter().product();
                            let tensor = Tensor::<f32> {
                                shape: output_shape,
                                data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                            };
                            self.add_output(tensor);
                        }
                        _ => {
                            panic!("Invalid inputs!");
                        }
                    }
                }
                else {
                    //formula W/2, H/2
                    match &self.inputs[0].tensor {
                        Some(v) => {
                            let w = v.shape[2] / 2;
                            let h = v.shape[3] / 2;
                            let output_shape = vec![v.shape[0], v.shape[1], w, h];
                            println!(
                                "Output tensor with shape {:?} created within Rust for operator {:?}!",
                                output_shape, self.op_type
                            );
                            let sz: usize = output_shape.iter().product();
                            let tensor = Tensor::<f32> {
                                shape: output_shape,
                                data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                            };
                            self.add_output(tensor);
                        }
                        _ => {
                            panic!("Invalid inputs!");
                        }
                    }
                }
                
            }

            OpType::EXP
            | OpType::ADD
            | OpType::SUBTRACT
            | OpType::DIVIDE
            | OpType::MULTIPLY
            | OpType::DROPOUT
            | OpType::ELU
            | OpType::RELU
            | OpType::GELU
            | OpType::SIGMOID
            | OpType::TANH
            | OpType::SCALAR_ADD
            | OpType::SCALAR_FLOORDIV
            | OpType::SCALAR_MULTIPLY
            | OpType::SCALAR_SUB
            | OpType::SCALAR_TRUEDIV
            | OpType::SOFTMAX 
            | OpType::BATCH_NORM 
            | OpType::LAYER_NORM 
            | OpType:: MULTIHEAD_ATTENTION //output shape of multiheaded attention is equal to shape of input "q" (first item in input list) when batch_first=True
             => {
                // let activations = vec![OpType::ELU, OpType::RELU, OpType::GELU, OpType::SIGMOID, OpType::TANH];
                // let inplace = if self.params.contains_key("inplace") {self.params["inplace"]=="True"} else {false};
                // if false && activations.iter().any(|&act| act==self.op_type)  {
                //        //TODO: implements inplace operator for activations
                // } else {
                match &self.inputs[0].tensor {
                    Some(v) => {
                        println!(
                            "Output tensor with shape {:?} created within Rust for operator {:?}!",
                            v.shape, self.op_type
                        );
                        let tensor = Tensor::<f32> {
                            shape: v.shape.clone(),
                            data_buffer: DataBuffer::CPUDataBuffer(vec![
                                0f32;
                                v.shape.iter().product()
                            ]),
                        };
                        self.add_output(tensor);
                    }
                    _ => {
                        panic!("Invalid inputs!");
                    }
                }
                // }
            }
            OpType::CONCAT => {
                //formula
                // w1 + w2 for axis 1
                // h1 + h2 for axis 0
                // let (mut batch, mut channel, mut w, mut h, mut batch_out, mut channel_out) = (0, 0, 0, 0, 0, 0);
                // let mut out_dim = 0;
                let mut output_shape = vec![0, 0, 0, 0];
                if let Some(v) = &mut self.inputs[0].tensor {
                    output_shape = v.shape.clone();
                }
                let idx = self.params["axis"].trim().parse::<usize>().unwrap();
                output_shape[idx] = 0;

                for tensor in &self.inputs {
                    if let Some(v) = &tensor.tensor {
                        output_shape[idx] += v.shape[idx];
                    }
                }

                println!(
                    "Output tensor with shape {:?} created within Rust for operator {:?}!",
                    output_shape, self.op_type
                );
                let sz: usize = output_shape.iter().product();
                let tensor = Tensor::<f32> {
                    shape: output_shape,
                    data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                };
                self.add_output(tensor);
            }
            OpType::SPLIT => {
                //formula
                // w/sizes for axis 1
                // h/sizes for axis 0
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let params = self.params.clone();
                        let idx = params["axis"].trim().parse::<usize>().unwrap();
                        assert!(idx < v.shape.len());
                        let mut output_shape = v.shape.clone();
                        if params.contains_key("sizes") {
                            let parts = params["sizes"].replace(['[',']'], "");
                            println!("{parts}");
                            for part in parts.split([',']) {
                                let size = part.trim().parse::<usize>().unwrap();
                                output_shape[idx] = size;
                                let sz: usize = output_shape.iter().product();
                                let tensor = Tensor::<f32> {
                                    shape: output_shape.clone(),
                                    data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                                };
                                self.add_output(tensor);
                                println!("Output tensor with shape {:?} created within Rust for operator {:?}!", output_shape, self.op_type);
                            }
                        } else {
                            panic!("Missing parameter 'sizes' for split!");
                        };
                    }
                    _ => {
                        panic!("Invalid tensor for axis {} split!", self.params["axis"]);
                    }
                }
            }
            OpType::FLAT => {
                //formula
                // output shape = [batch, channel * w * h]
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let sz: usize = v.shape.iter().product();
                        let output_shape = vec![v.shape[0], sz/v.shape[0]];
                        let tensor = Tensor::<f32> {
                            shape: output_shape.clone(),
                            data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                        };
                        self.add_output(tensor);
                        println!(
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

                        let output_shape : Vec<usize> = self.params["shape"].replace(['(', ')'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        let sz: usize = v.shape.iter().product();
                        let tensor = Tensor::<f32> {
                            shape: output_shape.clone(),
                            data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                        };
                        self.add_output(tensor);
                        println!(
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
                        assert!(self.params.contains_key("perm"));

                        let output_idx : Vec<usize> = self.params["perm"].replace(['(', ')'], "").split(",").map(|x| x.trim().parse::<usize>().unwrap()).collect();
                        let mut output_shape : Vec<usize>  = vec![];
                        for i in 0..v.shape.len() {
                            let idx = output_idx[i];
                            output_shape.extend([v.shape.get(idx).unwrap()]);
                        }
                        let sz: usize = v.shape.iter().product();
                        let tensor = Tensor::<f32> {
                            shape: output_shape.clone(),
                            data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                        };
                        self.add_output(tensor);
                        println!(
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
                        let tensor = Tensor::<f32> {
                            shape: output_shape.clone(),
                            data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; sz]),
                        };
                        self.add_output(tensor);
                        println!(
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

            OpType::LINEAR => {
                //formula
                // output shape = [batch, output dim]
                if !self.params.contains_key("out_dim") {
                    panic!("Missing important parameters!");
                }

                let output_dim = self.params["out_dim"].trim().parse::<usize>().unwrap();
                match &mut self.inputs[0].tensor {
                    Some(v) => {
                        let output_shape = vec![v.shape[0], output_dim];
                        let tensor = Tensor::<f32> {
                            shape: output_shape.clone(),
                            data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; v.shape[0] * output_dim]),
                        };
                        self.add_output(tensor);
                        println!(
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
        let name = self.op_type.as_str();
        let mut params = self.params.clone();
        if params.contains_key("kernel_w") && params.contains_key("kernel_h") {
            params.insert("kernel".to_string(), format!("[{}, {}]", params["kernel_w"], params["kernel_h"]));
            params.remove("kernel_w");
            params.remove("kernel_h");
        }

        if params.contains_key("stride_w") && params.contains_key("stride_h") {
            params.insert("stride".to_string(), format!("[{}, {}]", params["stride_w"], params["stride_h"]));
            params.remove("stride_w");
            params.remove("stride_h");
        }

        if params.contains_key("padding_w") && params.contains_key("padding_h") {
            params.insert("padding".to_string(), format!("[{}, {}]", params["padding_w"], params["padding_h"]));
            params.remove("padding_w");
            params.remove("padding_h");
        }
        params.remove("name");
        params.remove("tensors");
        params.remove("out_dim");
        params.remove("out_channels");
        params.remove("x"); //for add
        params.remove("y"); //for add

        params.remove("activation"); //fused activation not supported at the moment
        params.remove("inplace"); //inplace not supported at the moment
        params.remove("use_bias");//bias supported at the moment

        let mut input_names = "".to_string();
        let mut input_shapes = "".to_string();

        for input in &self.inputs {
            input_names += "%";
            input_names += input.name.as_str();
            input_names += ", ";

            input_shapes += input.get_ir().as_str();
            input_shapes += ", ";
        }

        input_names = input_names.trim().to_string();
        if &input_names[input_names.len()-1..] == "," {input_names.pop();}

        input_shapes = input_shapes.trim().to_string();
        if &input_shapes[input_shapes.len()-1..] == "," {input_shapes.pop();}

        let mut output_names = "".to_string();
        let mut output_shapes = "".to_string();

        for output in &self.outputs {
            output_names += "%";
            output_names += output.name.as_str();
            output_names += ", ";

            output_shapes += output.get_ir().as_str();
            output_shapes += ", ";
        }
        
        output_names = output_names.trim().to_string();
        if &output_names[output_names.len()-1..] == "," {output_names.pop();}

        output_shapes = output_shapes.trim().to_string();
        if &output_shapes[output_shapes.len()-1..] == "," {output_shapes.pop();}

        if params.is_empty() {
            let ir = format!("{output_names}=\"ufront.{name}\"({input_names}):({input_shapes}) -> {output_shapes}");
            ir
        }
        else {
            let ir = format!("{output_names}=\"ufront.{name}\"({input_names}){params:?}:({input_shapes}) -> {output_shapes}");
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
        println!("PyOperator::new");
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

    pub fn get_input_ndarray<'py>(&self, idx: usize, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        println!("Operator::get_input_ndarray idx {idx}");
        if self.raw_ptr > 0 && idx < self.num_of_inputs() {
            let operator = self.raw_ptr as *const Operator;
            unsafe {
                match (*operator).get_input(idx) {
                    Ok(tensor) => match &tensor.tensor {
                        Some(v) => match &v.data_buffer {
                            DataBuffer::CPUDataBuffer(data) => {
                                ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec())
                                    .unwrap()
                                    .into_pyarray(py)
                            }
                            _ => panic!("Tensor conversion failed!"),
                        },
                        _ => panic!("Tensor not initialized!"),
                    },
                    _ => panic!("Tensor not initialized!"),
                }
            }
        } else {
            panic!("Tensor not initialized or index out of bound!")
        }
    }

    pub fn get_output_ndarray<'py>(&self, idx: usize, py: Python<'py>) -> &'py PyArrayDyn<f32> {
        println!("Operator::get_output_ndarray {idx}");
        if self.raw_ptr > 0 && idx < self.num_of_outputs() {
            let operator = self.raw_ptr as *const Operator;
            unsafe {
                match (*operator).get_output(idx) {
                    Ok(tensor) => match &tensor.tensor {
                        Some(v) => match &v.data_buffer {
                            DataBuffer::CPUDataBuffer(data) => {
                                ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec())
                                    .unwrap()
                                    .into_pyarray(py)
                            }
                            _ => panic!("Tensor conversion failed!"),
                        },
                        _ => panic!("Tensor not initialized!"),
                    },
                    _ => panic!("Tensor not initialized!"),
                }
            }
        } else {
            panic!("Tensor not initialized or index out of bound!")
        }
    }

    pub fn add_input_ndarray(&mut self, x: PyReadonlyArrayDyn<f32>, name: String) {
        let x = x.as_array();
        match x.as_slice() {
            Some(v) => {
                let tensor = Tensor::<f32> {
                    shape: x.shape().to_vec(),
                    data_buffer: DataBuffer::CPUDataBuffer(v.to_vec()),
                };

                if self.raw_ptr > 0 {
                    unsafe {
                        let operator = std::mem::transmute::<u64, *mut Operator>(self.raw_ptr);
                        (*operator).add_input(tensor, name);
                    }
                } else {
                    println!("Input not added to the graph!");
                }

                println!(
                    "Tensor initialized with {:?} dimension within Rust",
                    x.shape().to_vec()
                );
            }
            _ => panic!("Invalid tensor inputs!"),
        }
    }

    pub fn add_input(&mut self, x: &TensorF32) -> PyResult<()> {
        println!("Operator::add_input");
        if self.raw_ptr > 0 {
            unsafe {
                let operator = std::mem::transmute::<u64, *mut Operator>(self.raw_ptr);
                match &x.tensor {
                    Some(v) => {
                        let tensor = Tensor::<f32> {
                            shape: v.shape.to_vec(),
                            data_buffer: v.data_buffer.clone(),
                        };
                        (*operator).add_input(tensor, x.name.clone());
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
    // pub fn get_input<'py>( &self, py: Python<'py>, idx : usize) -> Result<&'py TensorF32, ()> {
    //     if self.raw_ptr > 0 {
    //         let operator = self.raw_ptr as *const Operator;
    //         unsafe {

    //             match  (*operator).get_input(idx){
    //                 Ok(v) => {
    //                      Ok(v)
    //                 },
    //                 _ => {Err(())} //PyOSError::new_err("Invalid operator pointer!".to_string())
    //             }

    //         }
    //     }
    //     else { Err(())} //PyOSError::new_err("Invalid operator pointer!".to_string()
    // }

    pub fn get_output(&self, idx: usize) -> Py<TensorF32> {
        if self.raw_ptr > 0 {
            let operator = self.raw_ptr as *const Operator;
            unsafe {
                match (*operator).get_output(idx) {
                    Ok(tensorf32) => match &tensorf32.tensor {
                        Some(v) => {
                            let tensor = Tensor::<f32> {
                                shape: v.shape.to_vec(),
                                data_buffer: v.data_buffer.clone(),
                            };

                            Python::with_gil(|py| {
                                Py::new(
                                    py,
                                    TensorF32 {
                                        tensor: Some(tensor),
                                        name : tensorf32.name.clone(),
                                    },
                                )
                                .unwrap()
                            })
                        }
                        _ => {
                            panic!("Unable to obtain output!");
                        }
                    },
               
                    _ => {
                        panic!("Unable to obtain output!");
                    }
                
                }
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
