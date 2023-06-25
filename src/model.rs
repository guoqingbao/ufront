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

use core::panic;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use tuple_conv::RepeatedTuple;
// use pyo3::types::PyList;
// use pyo3::wrap_pyfunction;
// use pyo3_log;
// use crate::error::RustError;
use crate::databuffer::DataBuffer;
use crate::databuffer::Buffer;
use crate::graph::Graph;
use crate::graph::GraphTrait;
use crate::operator::Operator;
use crate::operator::PyOperator;
// use crate::prelude::Tensor;
use crate::tensor::Tensor;
use crate::tensor::TensorU;
use crate::types::DataType;
use crate::types::OpType;
use crate::types::WeightType;
use indexmap::IndexMap;
use log::{info, warn, error};
use crate::optimizer::Optimizer;

use std::collections::HashMap;
use std::convert::TryFrom;
use std::rc::Rc;
use std::sync::Once;
use half::f16;

static START: Once = Once::new();

use rawapi;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;

pub trait FunctionTrait {
    fn input(&self);
    fn output(&self);

    fn conv2d(&self);
    fn matmul(&self);
    fn maxpool2d(&self);
    fn avgpool2d(&self);
    fn batchnorm(&self);
    fn linear(&self);
    fn flat(&self);
    fn multihead_attention(&self);
    fn layer_norm(&self);
    fn embedding(&self);
    fn expand(&self);
    fn softmax(&self);
    fn transpose(&self);
    fn reshape(&self);
    fn dropout(&self);

    fn add(&self);
    fn subtract(&self);
    fn multiply(&self);
    fn divide(&self);
    fn pow(&self);
    fn exp(&self);
    fn sin(&self);
    fn cos(&self);
    fn mean(&self);
    fn reverse(&self);
    fn rsqrt(&self);
    fn floor_divide(&self);

    fn mseloss(&self);
    fn gelu(&self);
    fn relu(&self);
    fn tanh(&self);
    fn elu(&self);
    fn sigmoid(&self);
    fn concat(&self);
    fn split(&self);

    fn sadd(&self);
    fn ssub(&self);
    fn smultiply(&self);
    fn sfloordiv(&self);
    fn struediv(&self);

    fn type_as(&self);
    fn view(&self);
    fn to(&self);
    fn unsqueeze(&self);
    fn permute(&self);
    fn contigeous(&self);
    fn identity(&self);
    fn attribute(&self);

    fn getitem(&self);
    fn getattr(&self);

    fn float(&self);
}
pub trait ModelTrait {
    fn compile(&self);
    fn forward(&self);
    fn backward(&self);
    fn update(&self);
    fn zero_gradients(&self);
}

#[pyclass]
pub struct Model {
    pub graph: Graph,
    #[pyo3(get, set)]
    pub optimizer: Optimizer,
    args: IndexMap<String, String>,
    argshapes: IndexMap<String, String>,
    ssa_ids: HashMap<String, i32>,
    op_names: Vec<String>,
    op_idx : i32,
    #[pyo3(get, set)]
    pub weight_type: WeightType,
}

#[pymethods]
impl Model {
    #[new]
    pub fn new() -> Model {
        START.call_once(|| {
            env_logger::init();
        });
        info!("Model::new");
        let params = HashMap::from([
            ("type".to_string(), "sgd".to_string()),
            ("lr".to_string(), "0.01".to_string()),
            ("momentum".to_string(), "0".to_string()),
            ("nesterov".to_string(), "False".to_string()),
            ("weight_decay".to_string(), "0".to_string()),
        ]);
        Model {
            graph: Graph { operators: vec![] },
            optimizer: Optimizer {
                // optim_type: OptimizerType::SGD,
                params,
            },
            args: IndexMap::new(),
            argshapes: IndexMap::new(),
            ssa_ids: HashMap::new(),
            op_names: Vec::new(),
            op_idx: 0,
            weight_type: WeightType::INTERNAL,
        }
    }

    fn build_ssa_ids(&mut self) {
        let mut idx = 1;
        for operator in &self.graph.operators {
            for output in &operator.outputs {
                if !self.ssa_ids.contains_key(&output.name) {
                    self.ssa_ids.insert(output.name.clone(), idx);
                    idx += 1;
                }
            }
        }
    }

    #[getter]
    pub fn get_ssa_ids<'py>(&self, py: Python<'py>) -> &'py PyDict {
        self.ssa_ids.clone().into_py_dict(py)
    }

    #[pyo3(signature = (**kwds))]
    pub fn compile(&mut self, kwds: Option<&PyDict>) {
        info!("Model::compile");

        match kwds {
            Some(args) => {
                info!("\r\n{args:?}");
            }
            _ => {
                panic!("No arguments for compile!");
            }
        }

        self.graph.compile(kwds);
        self.build_ssa_ids();
    }

    pub fn forward(&self) {
        info!("Model::forward");
        self.graph.forward();
    }

    pub fn backward(&self) {
        info!("Model::backward");
        self.graph.backward();
    }

    pub fn update(&self) {
        info!("Model::update");
        self.graph.update();
    }

    pub fn zero_gradients(&self) {
        info!("Model::zero_gradients");
        self.graph.zero_gradients();
    }

    #[pyo3(text_signature = "($self, pyop)")]
    pub fn add_operator(&mut self, pyop: &mut PyOperator) {
        if !pyop.params.contains_key("name")
            || pyop.params["name"].is_empty()
            || self.op_names.contains(&pyop.params["name"])
        {
            let mut name: String =
                if pyop.params.contains_key("name") && !pyop.params["name"].is_empty() {
                    pyop.params["name"].clone()
                } else {
                    pyop.op_type.as_str().to_string()
                };
            for opname in self.op_names.iter().rev() {
                if opname.find(&name) != None {
                    match opname.find("_") {
                        Some(pos) => {
                            name += "_";
                            name += (opname[pos + 1..].parse::<usize>().unwrap() + 1)
                                .to_string()
                                .as_str();
                        }
                        _ => {
                            name += "_1";
                        }
                    }
                    break;
                }
            }
            pyop.params.remove("name");
            pyop.params.insert("name".to_string(), name.clone());
            self.op_names.push(name);
        }

        let mut logstr = format!("Op: {}, ", pyop.op_type.as_str());
        let mut idxmap = IndexMap::<String, String>::new();
        for (key, value) in &pyop.params {
            if key=="initializer" {
                logstr += format!("{key}:\"__elided__\", ").as_str();
            }
            else {
                logstr += format!("{key}:{value}, ").as_str();
            }
            idxmap.insert(key.to_string(), value.to_string());
        }
        info!("{}", logstr);

        let operator = Box::new(Operator::new(pyop.op_type, idxmap));

        let handle_ptr: *const Operator = &*operator;

        self.graph.operators.push(operator);

        // let ptr = &mut self.graph.operators[idx];
        pyop.raw_ptr = handle_ptr as u64;
        // return Python::with_gil(|py| -> PyResult<Py<PyOperator>> {
        //     let foo: Py<PyOperator> = Py::new(py, PyOperator {id: id, op_type : optype})?;
        //     return Ok(foo)
        // });

        //
        self.op_idx += 1;
        // return Err(PyOSError::new_err("Failed to create operator!".to_string()));
    }

    #[pyo3(text_signature = "($self, pyop)")]
    pub fn remove_operator(&mut self, pyop: &mut PyOperator) {
        // let idx = self.num_of_operators();
        // let ptr = pyop.raw_ptr as *const Operator;

        // unsafe {
        self.graph
            .operators
            .retain(|x| &**x as *const Operator as u64 != pyop.raw_ptr);
        // }

        info!(
            "Op: {}, ptr: {} removed from computation graph!",
            pyop.op_type.as_str(),
            pyop.raw_ptr
        );
        pyop.raw_ptr = 0;
    }
    // #[pyo3(text_signature = "($self, optype, args)")]
    // pub fn op(&mut self, optype:OpType, args: HashMap<String, String>) -> PyResult<Py<PyOperator>> {
    //     // let _optype = OpType::try_from(optype as u32);
    //     let id = self.num_of_operators();

    //     print!("Op: {}, ", optype.as_str());
    //     for (key, value) in &args {
    //         print!("{}:{}, ", key, value)
    //     }
    //     print!("\n");
    //     // let mut params_ : HashMap<String, String> = HashMap::new();
    //     // params_.extend(args.into_iter());
    //     let operator = Box::new(Operator::new(self.num_of_operators(), optype, args));

    //     self.graph.operators.push(operator);
    //     return Python::with_gil(|py| -> PyResult<Py<PyOperator>> {
    //         let foo: Py<PyOperator> = Py::new(py, PyOperator {id: id, op_type : optype})?;
    //         return Ok(foo)
    //     });

    //     //

    //     // return Err(PyOSError::new_err("Failed to create operator!".to_string()));
    // }

    pub fn num_of_operators(&self) -> usize {
        self.graph.operators.len()
    }

    pub fn dump_tosa_ir(&self) -> String {
        let ufront_ir = self.dump_ir();
        let rust_cstring = CString::new(ufront_ir).unwrap();
        let rust_cstring = rust_cstring.as_bytes_with_nul().as_ptr() as *const c_char;

        let tosa_ir_cstr = unsafe { rawapi::ufront_to_tosa(rust_cstring) };
        let cstr = unsafe { CStr::from_ptr(tosa_ir_cstr).to_str().unwrap() };
        String::from(cstr)
    }

    pub fn dump_ir(&self) -> String {
        // info!("{:?}", self.argshapes);
        let mut argstr = "".to_string();
        for arg in self.argshapes.keys() {
            argstr += "%";
            argstr += arg;
            argstr += ": ";
            argstr += self.argshapes[arg].as_str();
            argstr += ", ";
        }
        
        argstr = argstr.trim().to_string();
        if &argstr[argstr.len() - 1..] == "," {
            argstr.pop();
        }

        let mut output_shapes = "".to_string();
        let mut last_outname = "".to_string();

        if let Some(op) = &self.graph.operators.last() {
            for output in &op.outputs {
                output_shapes += output.get_ir().as_str();
                output_shapes += ", ";

                last_outname += "%";

                if self.ssa_ids.contains_key(&output.name) {
                    last_outname += &self.ssa_ids[&output.name].to_string();
                } else {
                    last_outname += output.name.as_str();
                }

                last_outname += ", ";
            }
        }
        output_shapes = output_shapes.trim().to_string();
        if &output_shapes[output_shapes.len() - 1..] == "," {
            output_shapes.pop();
        }

        last_outname = last_outname.trim().to_string();
        if &last_outname[last_outname.len() - 1..] == "," {
            last_outname.pop();
        }

        let header = format! {"func.func @forward({argstr}) -> {output_shapes} "};

        // info!("{:?}", self.args);
        let mut op_ir = "".to_string();
        for op in &self.graph.operators {
            op_ir += "\t";
            op_ir += op.dump_ir(Some(&self.ssa_ids)).as_str();
            op_ir += "\n";
        }

        format!("{header} {{ \n{op_ir}\treturn {last_outname}: {output_shapes}\n}}")
    }
    // pub fn num_of_op_inputs(&self, id : usize) -> usize {
    //     for i in 0..self.graph.operators.len() {
    //         if self.graph.operators[i].id == id {
    //             return self.graph.operators[i].num_of_inputs();
    //         }
    //     }
    //     return 0
    // }

    // pub fn num_of_op_outputs(&self, id : usize) -> usize {
    //     for i in 0..self.graph.operators.len() {
    //         if self.graph.operators[i].id == id {
    //             return self.graph.operators[i].num_of_outputs();
    //         }
    //     }
    //     return 0
    // }

    // pub fn get_op_input(&self, id : usize, idx : usize) {
    //     for i in 0..self.graph.operators.len() {
    //         if self.graph.operators[i].id == id {
    //             if idx < self.graph.operators[i].num_of_inputs() {
    //                 return self.graph.operators[i].get_input(idx)
    //             }
    //         }
    //     }
    //     // return None;
    // }

    // pub fn get_op_output(&self, id : usize, idx : usize) {
    //     for i in 0..self.graph.operators.len() {
    //         if self.graph.operators[i].id == id {
    //             if idx < self.graph.operators[i].num_of_outputs() {
    //                 return self.graph.operators[i].get_output(idx)
    //             }
    //         }
    //     }
    //     // return None;
    // }

    // }

    #[pyo3(text_signature = "($self, op_type, args)")]
    pub fn add_layer(&mut self, op_type: OpType, args: Option<&PyDict>) -> Py<PyOperator> {
        // let mut params = HashMap::<String, String>::new();
        let mut op = PyOperator {
            op_type,
            params: HashMap::<String, String>::new(),
            raw_ptr: 0,
        };
        // let mut argstr = "".to_string();
        match args {
            Some(para) => {
                for key in para.keys() {
                    if key.to_string() == "np_tensor" 
                    || key.to_string() == "weight" || key.to_string() == "bias" 
                    || key.to_string() == "q" || key.to_string() == "k" || key.to_string() == "v" 
                    || key.to_string() == "x" || key.to_string() == "y" || key.to_string() == "tensors" 
                    || key.to_string() == "weight_q" || key.to_string() == "weight_k" || key.to_string() == "weight_v" || key.to_string() == "weight_o"  
                    || key.to_string() == "bias_q" || key.to_string() == "bias_k" || key.to_string() == "bias_v" || key.to_string() == "bias_o"  
                    {
                        // println!("Ignored {key}");
                    }
                    else if key.to_string() != "input" {
                        op.params
                            .insert(key.to_string(), para.get_item(key).unwrap().to_string());
                    } 
                    // else {
                    //     argstr = para.get_item(key).unwrap().to_string();
                    // }
                }

                if op_type == OpType::INPUT {
                    if para.contains("tensors").unwrap() {
                        let ret = para
                            .get_item("tensors")
                            .unwrap()
                            .extract::<Vec<PyRef<Tensor>>>(); // .downcast::<Tensor>();
                        match ret {
                            Ok(vlist) => {
                                for v in vlist {
                                    // self.args.insert(v.name.clone(), argstr.clone());
                                    self.argshapes.insert(v.name.clone(), v.get_ir());
                                }
                                info!("Input: {:?}!", self.argshapes);
                            }
                            _ => {
                                panic! {"Not valid inputs!"};
                            }
                        }
                    } else {
                        panic! {"Missing important arguments (tensors?)"};
                    }
                } else {
                    self.add_operator(&mut op);
                }

                if op_type == OpType::CONCAT {
                    if para.contains("tensors").unwrap() {
                        let ret = para
                            .get_item("tensors")
                            .unwrap()
                            .extract::<Vec<PyRef<Tensor>>>(); // .downcast::<Tensor>();
                        match ret {
                            Ok(vlist) => {
                                for v in vlist {
                                    if op.add_input(&v).is_ok() {
                                        info!(
                                            "A list of input tensors added for operator {op_type:?}!"
                                        );
                                    }
                                }
                                op.calculate_output();
                            }
                            _ => {
                                panic! {"Not a valid input type!"};
                            }
                        }
                    } else {
                        panic! {"Missing important arguments (tensors?)"};
                    }
                } else if op_type == OpType::ADD || op_type == OpType::MULTIPLY || op_type == OpType::SUBTRACT || op_type == OpType::MATMUL ||
                        op_type == OpType::BATCH_MATMUL || op_type == OpType::LESS || op_type == OpType::AND ||
                        ((op_type == OpType::SLICE) && !para.contains("input").unwrap() )    {
                    if para.contains("x").unwrap() && para.contains("y").unwrap() {
                        let x = para.get_item("x").unwrap().extract::<PyRef<Tensor>>(); // .downcast::<Tensor>();
                        let y = para.get_item("y").unwrap().extract::<PyRef<Tensor>>(); // .downcast::<Tensor>();
                        match (x, y) {
                            (Ok(v1), Ok(v2)) => {
                                if op.add_input(&v1).is_ok() && op.add_input(&v2).is_ok() {
                                    op.calculate_output();
                                }
                            }
                            _ => {
                                panic! {"Not a valid input type!"};
                            }
                        }
                    } else {
                        panic! {"Missing important arguments (x, or y?)"};
                    }
                } else if op_type == OpType::MULTIHEAD_ATTENTION {
                    if para.contains("q").unwrap()
                        && para.contains("k").unwrap()
                        && para.contains("v").unwrap()
                    {
                        let q = para.get_item("q").unwrap().extract::<PyRef<Tensor>>(); 
                        let k = para.get_item("k").unwrap().extract::<PyRef<Tensor>>(); 
                        let v = para.get_item("v").unwrap().extract::<PyRef<Tensor>>(); 
                        
                        // if self.argshapes.len() == 0 {
                        //     match &q {
                        //         Ok(v) => {
                        //             if v.name.find("input") == Some(0) || v.name.find("x") == Some(0) {
                        //                 self.args.insert(v.name.clone(), argstr.clone());
                        //                 self.argshapes.insert(v.name.clone(), v.get_ir());
                        //                 info!("Multihead Attention inputs come from forward input!");
                        //             }
                        //         }
                        //         _ => {}
                        //     }
                        // }

                        match (q, k, v) {
                            (Ok(q1), Ok(k1), Ok(v1)) => {
                                if op.add_input(&q1).is_ok()
                                    && op.add_input(&k1).is_ok()
                                    && op.add_input(&v1).is_ok()
                                {
                                    if para.contains("weight_q").unwrap()
                                    && para.contains("weight_k").unwrap()
                                    && para.contains("weight_v").unwrap()
                                    && para.contains("bias_q").unwrap()
                                    && para.contains("bias_k").unwrap()
                                    && para.contains("bias_v").unwrap()
                                    && para.contains("weight_o").unwrap()
                                    && para.contains("bias_o").unwrap()

                                    {
                                        let weight_q = para.get_item("weight_q").unwrap().extract::<PyRef<Tensor>>(); 
                                        let weight_k = para.get_item("weight_k").unwrap().extract::<PyRef<Tensor>>(); 
                                        let weight_v = para.get_item("weight_v").unwrap().extract::<PyRef<Tensor>>(); 
                                        let bias_q = para.get_item("bias_q").unwrap().extract::<PyRef<Tensor>>(); 
                                        let bias_k = para.get_item("bias_k").unwrap().extract::<PyRef<Tensor>>(); 
                                        let bias_v = para.get_item("bias_v").unwrap().extract::<PyRef<Tensor>>(); 

                                        let weight_o = para.get_item("weight_o").unwrap().extract::<PyRef<Tensor>>(); 
                                        let bias_o = para.get_item("bias_o").unwrap().extract::<PyRef<Tensor>>(); 

                                        match (weight_q, weight_k, weight_v, bias_q, bias_k, bias_v, weight_o, bias_o) {
                                            (Ok(wq), Ok(wk), Ok(wv), Ok(bq), Ok(bk), Ok(bv), Ok(wo), Ok(bo)) => {
                                                if op.add_input(&wq).is_ok()
                                                && op.add_input(&wk).is_ok()
                                                && op.add_input(&wv).is_ok()
                                                && op.add_input(&bq).is_ok()
                                                && op.add_input(&bk).is_ok()
                                                && op.add_input(&bv).is_ok()
                                                && op.add_input(&wo).is_ok()
                                                && op.add_input(&bo).is_ok()
                                                {
                                                    op.calculate_output();
                                                }
                                            }
                                            _ => {
                                                panic! {"Not the valid weight type!"};
                                            }
                                        }
                                    }
                                    else {
                                        op.calculate_output();
                                    }
                                }
                            }
                            _ => {
                                panic! {"Not a valid input type!"};
                            }
                        }
                    }
                    else {
                        panic! {"Missing important arguments (q, k, or v?)"};
                    }
                } else if op_type == OpType::PARAMETER || op_type == OpType::TENSOR {

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
                                if op.add_input(&tensor).is_ok() {
                                    op.calculate_output();
                                }
                            }
                            _ => {panic! {"Invalid tensor argument!"};}
                        }

                        // if DataType::Float == dtype {
                        //     let np_tensor = para.get_item("np_tensor").unwrap().extract::<PyReadonlyArrayDyn<f32>>();
                        //     match np_tensor {
                        //         Ok(v) => {
                        //             tensor.set_ndarray(v);
                        //             if op.add_input(&tensor).is_ok() {
                        //                 op.calculate_output();
                        //             }
                        //         }
                        //         _ => {panic! {"Invalid tensor argument!"};}
                        //     }
                        // } else if DataType::Half == dtype {
                        //     let np_tensor = para.get_item("np_tensor").unwrap().extract::<PyReadonlyArrayDyn<f16>>();
                        //     match np_tensor {
                        //         Ok(v) => {
                        //             tensor.set_ndarrayf16(v);
                        //             if op.add_input(&tensor).is_ok() {
                        //                 op.calculate_output();
                        //             }
                        //         }
                        //         _ => {panic! {"Invalid tensor argument!"};}
                        //     }
                        // } else {
                        //     panic! {"Not a valid tensor type!"};
                        // }

                    } else {
                        panic! {"Missing important arguments ('np_tensor', 'dtype' and 'name')!"};
                    }

                    // if para.contains("np_tensor").unwrap() && para.contains("dtype").unwrap() {
                    //     let np_tensor = para
                    //         .get_item("np_tensor")
                    //         .unwrap()
                    //         .extract::<PyReadonlyArrayDyn<f32>>();
                    //     let dtype = para
                    //         .get_item("dtype")
                    //         .unwrap()
                    //         .extract::<DataType>()
                    //         .unwrap();
                    //     match dtype {
                    //         DataType::Float
                    //         | DataType::Double
                    //         | DataType::Int32
                    //         | DataType::Int64
                    //         | DataType::Bool => {}
                    //         _ => {
                    //             panic! {"Not supported type!"};
                    //         }
                    //     }

                    //     match np_tensor {
                    //         Ok(_np_tensor) => {
                    //             let mut tensor = Tensor {
                    //                 tensor: None,
                    //                 name: para.get_item("name").unwrap().to_string(),
                    //                 dtype: dtype,
                    //             };
                    //             tensor.set_ndarray(_np_tensor);
                    //             if op.add_input(&tensor).is_ok() {
                    //                 op.calculate_output();
                    //             }
                    //         }
                    //         _ => {
                    //             panic! {"Not a valid input type!"};
                    //         }
                    //     }
                    // } else {
                    //     panic! {"Missing important arguments (q, k, or v?)"};
                    // }
                } else if op_type == OpType::MASKEDFILL {
                    if para.contains("input").unwrap()
                        && para.contains("mask").unwrap()
                        && para.contains("value").unwrap()
                    {
                        let input = para.get_item("input").unwrap().extract::<PyRef<Tensor>>(); // .downcast::<Tensor>();
                        let mask = para.get_item("mask").unwrap().extract::<PyRef<Tensor>>(); // .downcast::<Tensor>();

                        match (input, mask) {
                            (Ok(input1), Ok(mask1)) => {
                                if op.add_input(&input1).is_ok()
                                    && op.add_input(&mask1).is_ok()
                                {
                                    op.calculate_output();
                                }
                            }
                            _ => {
                                panic! {"Not a valid input type!"};
                            }
                        }
                    } else {
                        panic! {"Missing important arguments (q, k, or v?)"};
                    }
                } else if (op_type == OpType::BATCH_NORM || op_type == OpType::LAYER_NORM)&& para.contains("input").unwrap() {
                    let ret = para.get_item("input").unwrap().extract::<PyRef<Tensor>>(); 
                    match ret {
                        Ok(v) => {
                            // if self.argshapes.len() == 0 && (v.name.find("input") == Some(0) || v.name.find("x") == Some(0)) {
                            //     self.args.insert(v.name.clone(), argstr.clone());
                            //     self.argshapes.insert(v.name.clone(), v.get_ir());
                            // }
                            if op.add_input(&v).is_ok() {
                                if para.contains("weight").unwrap() && para.contains("bias").unwrap()  {
                                    let ret1 = para.get_item("weight").unwrap().extract::<PyRef<Tensor>>(); 
                                    let ret2 = para.get_item("bias").unwrap().extract::<PyRef<Tensor>>(); 

                                    match (ret1, ret2) {
                                        (Ok(v1), Ok(v2)) => {
                                            if op.add_input(&v1).is_ok() && op.add_input(&v2).is_ok() {
                                                if para.contains("mean").unwrap() && para.contains("variance").unwrap() {
                                                    let ret3 = para.get_item("mean").unwrap().extract::<PyRef<Tensor>>(); 
                                                    let ret4 = para.get_item("variance").unwrap().extract::<PyRef<Tensor>>(); 
                                                    match (ret3, ret4) {
                                                        (Ok(v3), Ok(v4)) => {
                                                            if op.add_input(&v3).is_ok() && op.add_input(&v4).is_ok() {
                                                                op.calculate_output();
                                                            }
                                                        }
                                                        _ => {
                                                            panic! {"Not the valid mean and variance type!"};
                                                        }
                                                    }
                                                } else {
                                                    op.calculate_output();
                                                }
                                            }
                                        }
                                        _ => {
                                            panic! {"Not the valid weight and bias type!"};
                                        }
                                    }
                                } else {
                                    op.calculate_output();
                                }
                            }
                        }
                        _ => {
                            panic! {"Not a valid input type!"};
                        }
                    }
                } else if para.contains("input").unwrap()  {
                        let ret = para.get_item("input").unwrap().extract::<PyRef<Tensor>>(); 
                        match ret {
                            Ok(v) => {
                                // if v.name.find("input") == Some(0) || v.name.find("x") == Some(0) {
                                //     self.args.insert(v.name.clone(), argstr.clone());
                                //     self.argshapes.insert(v.name.clone(), v.get_ir());
                                // }
                                if op.add_input(&v).is_ok() {
                                    if para.contains("weight").unwrap()  {
                                        let ret = para.get_item("weight").unwrap().extract::<PyRef<Tensor>>();
                                        match ret {
                                            Ok(v) => {
                                                if op.add_input(&v).is_ok() {
                                                    if para.contains("bias").unwrap() {
                                                        let ret1 = para.get_item("bias").unwrap().extract::<PyRef<Tensor>>();
                                                        match ret1 {
                                                            Ok(v1) => {
                                                                if op.add_input(&v1).is_ok() {
                                                                    op.calculate_output();
                                                                }
                                                            }
                                                            _ => {
                                                                panic! {"Not a valid bias type!"};
                                                            }
                                                        }
                                                    } else {
                                                        op.calculate_output();
                                                    }
                                                }
                                            }
                                            _ => {
                                                panic! {"Not a valid weight type!"};
                                            }
                                        }
                                    } else {
                                        op.calculate_output();
                                    }
                                }
                            }
                            _ => {
                                panic! {"Not a valid input type!"};
                            }
                        }
                } else if op_type == OpType::ARANGE { //arange has no inputs
                    op.calculate_output();
                }

                Python::with_gil(|py| Py::new(py, op).unwrap())
            }
            _ => {
                panic!("No arguments for operator {:?}", op_type);
            }
        }

        // return &op;
    }
    // impl FunctionTrait for Model {

    pub fn create_tensor(
        &mut self,
        shape: Vec<usize>,
        dtype: DataType,
        requires_grad: bool,
        name: String,
    ) -> Py<Tensor> {
        // let sp = shape.as_slice().to_vec();

        let shape = shape.to_vec();
        let tensor = match dtype {
            DataType::Float => {
                Some(TensorU {
                    shape: shape.clone(),
                    data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(shape.iter().product(), Some(vec![0f32; shape.iter().product()]))),
                })
            }
            DataType::Half => {
                Some(TensorU {
                    shape: shape.clone(),
                    data_buffer: DataBuffer::CPUDataBuffer(Buffer::new(shape.iter().product(), Some(vec![half::f16::ZERO; shape.iter().product()]))),
                })
            }
            _ => {
                panic!("Not supported type at the moment!");
            }
        };
        info!("Tensor initialized with {shape:?} dimension within Rust");
        Python::with_gil(|py| {
            Py::new(
                py,
                Tensor {
                    tensor,
                    name: name.clone(),
                    dtype: dtype,
                },
            )
            .unwrap()
        })

       
    }

    #[pyo3(signature = (**kwds))]
    pub fn parameter(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::PARAMETER, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn tensor(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::TENSOR, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn call(&mut self, py: Python, kwds: Option<&PyDict>) -> Py<PyOperator> {
        // self.handle_operator(OpType::CALL, kwds)
        info!("{:?}\r\n", kwds);
        let op_type = OpType::CALL;
        let mut op = PyOperator {
            op_type,
            params: HashMap::<String, String>::new(),
            raw_ptr: 0,
        };
        // let mut argstr = "".to_string();
        match kwds {
            Some(para) => {
                for key in para.keys() {
                    if key.to_string() != "input" {
                        op.params
                            .insert(key.to_string(), para.get_item(key).unwrap().to_string());
                    } 
                    // else {
                    //     argstr = para.get_item(key).unwrap().to_string();
                    // }
                }

                let callback = para.get_item("callback").unwrap().extract::<PyObject>();
                let func_args = para.get_item("args").unwrap().extract::<&PyDict>(); // .downcast::<Tensor>();
                self.add_operator(&mut op);

                match (callback, func_args) {
                    (Ok(_callback), Ok(_func_args)) => {
                        info!("Function args  {:?}", _func_args);

                        for key in _func_args.keys() {
                            let x = _func_args
                                .get_item(key)
                                .unwrap()
                                .extract::<PyRef<Tensor>>();
                            match x {
                                Ok(_x) => {
                                    if op.add_input(&_x).is_ok() {
                                        info!(
                                            "A list of input tensors added for operator {op_type:?}!"
                                        );
                                    }
                                }
                                _ => {
                                    info!("{:?} is not tensor argument!", key);
                                }
                            }
                        }

                        match _callback.call(py, (), kwds) {
                            Ok(ret) => {
                                // if para.contains("return").is_ok() {
                                info!("Model::add_layer calculate output for {:?} by calling the external function {}", 
                                            op_type, para.get_item("func").unwrap().to_string());
                                let tensor = ret.extract::<PyRef<Tensor>>(py);

                                match tensor {
                                    Ok(v) => {
                                        op.add_output(&v);
                                        info!("Output tensor with shape {:?} created within Rust for operator {:?}!", v.get_shape(py), op_type);
                                    }
                                    // _ => panic!("Invalid tensor outputs!"),
                                    _ => {
                                        info!("The return value from function call {} is not tensor output!", para.get_item("func").unwrap().to_string());
                                    }
                                }
                            }
                            Err(e) => {
                                error!("No return values of the callback! {:?}", e);
                            }
                        }
                    }
                    _ => {
                        panic!("Invalid call!");
                    }
                }
            }
            _ => {
                panic!("Invalid call!");
            }
        }

        Python::with_gil(|py| Py::new(py, op).unwrap())
    }

    #[pyo3(signature = (**kwds))]
    pub fn slice_tensor(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SLICE, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn input(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::INPUT, kwds)
    }

    pub fn output(&self) {}

    #[pyo3(signature = (**kwds))]
    pub fn eq(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::EQ, kwds)
    }

    fn handle_operator(&mut self, op_type: OpType, kwds: Option<&PyDict>) -> Py<PyOperator> {
        match kwds {
            Some(args) => {
                if op_type != OpType::TENSOR && op_type != OpType::PARAMETER {
                    info!("\r\n{args:?}");
                }
                else {
                    let mut mp = HashMap::<String, String>::new();
                    for key in args.keys() {
                        if key.to_string() == "initializer" {
                            mp.insert(key.to_string(), "__elided__".to_string());
                        } else {
                            mp.insert(key.to_string(), args.get_item(key).unwrap().to_string());
                        }
                    }
                    info!("\r\n{mp:?}");
                }
                self.add_layer(op_type, kwds)
            }
            _ => {
                panic!("No arguments for {:?}!", op_type);
            }
        }
    }
    // #[pyfunction(kwds="**")]
    #[pyo3(signature = (**kwds))]
    pub fn conv2d(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::CONV2D, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn batch_matmul(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::BATCH_MATMUL, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn matmul(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::MATMUL, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn pool2d(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::POOL2D, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn batch_norm(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::BATCH_NORM, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn linear(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::LINEAR, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn dense(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::LINEAR, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn flat(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::FLAT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn multihead_attention(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::MULTIHEAD_ATTENTION, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn layer_norm(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::LAYER_NORM, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn embedding(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::EMBEDDING, kwds)
    }
    
    #[pyo3(signature = (**kwds))]
    pub fn expand(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::EXPAND, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn softmax(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SOFTMAX, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn transpose(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::TRANSPOSE, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn reshape(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::RESHAPE, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn dropout(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::DROPOUT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn add(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::ADD, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn subtract(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SUBTRACT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn multiply(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::MULTIPLY, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn divide(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::DIVIDE, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn pow(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::POW, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn exp(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::EXP, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn sin(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SIN, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn cos(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::COS, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn mean(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::MEAN, kwds)
    }

    pub fn reverse(&self) {}

    #[pyo3(signature = (**kwds))]
    pub fn rsqrt(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::RSQRT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn floor_divide(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::FLOOR_DIVIDE, kwds)
    }

    pub fn mseloss(&self) {}

    #[pyo3(signature = (**kwds))]
    pub fn gelu(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::GELU, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn relu(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::RELU, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn tanh(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::TANH, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn elu(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::ELU, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn sigmoid(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SIGMOID, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn hardswish(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::HARDSWISH, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn hardsigmoid(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::HARDSIGMOID, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn silu(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SILU, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn concat(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::CONCAT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn split(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SPLIT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn chunk(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::CHUNK, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn sadd(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SCALAR_ADD, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn ssub(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SCALAR_SUB, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn smultiply(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SCALAR_MULTIPLY, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn sfloordiv(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SCALAR_FLOORDIV, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn struediv(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SCALAR_TRUEDIV, kwds)
    }

    pub fn type_as(&self) {}

    pub fn view(&self) {}

    pub fn to(&self) {}

    pub fn unsqueeze(&self) {}

    pub fn permute(&self) {}

    pub fn contigeous(&self) {}

    #[pyo3(signature = (**kwds))]
    pub fn identity(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::IDENTITY, kwds)
    }

    pub fn attribute(&self) {}

    pub fn getitem(&self) {}

    pub fn getattr(&self) {}

    #[pyo3(signature = (**kwds))]
    pub fn float(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::FLOAT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn bool(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::BOOL, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn invert(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::INVERT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn And(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::AND, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn detach(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::DETACH, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn cumsum(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::CUMSUM, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn arange(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::ARANGE, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn masked_fill(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::MASKEDFILL, kwds)
    }


    #[pyo3(signature = (**kwds))]
    pub fn repeat(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::REPEAT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn uniform_like(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::UNIFORM_LIKE, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn less(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::LESS, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn cast(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator>  {
        self.handle_operator(OpType::CAST, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn clip(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::CLIP, kwds)
    }
}
