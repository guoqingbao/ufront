use core::panic;
use pyo3::prelude::*;
use pyo3::types::PyDict;
// use pyo3::types::PyList;
// use pyo3::wrap_pyfunction;
// use pyo3_log;
// use crate::error::RustError;

use crate::graph::Graph;
use crate::graph::GraphTrait;
use crate::operator::Operator;
use crate::operator::PyOperator;
use crate::tensor::TensorF32;
use crate::types::{OpType};

use crate::optimizer::Optimizer;

use std::collections::HashMap;
use std::convert::TryFrom;
use std::rc::Rc;

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

    fn init_param(&self);
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
}

#[pymethods]
impl Model {
    #[new]
    pub fn new() -> Model {
        println!("Model::new");
        let params = HashMap::from([("type".to_string(),"sgd".to_string()), 
                                ("lr".to_string(),"0.01".to_string()), 
                                ("momentum".to_string(),"0".to_string()), 
                                ("nesterov".to_string(),"False".to_string()), 
                                ("weight_decay".to_string(),"0".to_string())]);
        Model {
            graph: Graph { operators: vec![] },
            optimizer: Optimizer {
                // optim_type: OptimizerType::SGD,
                params: params,
            },
        }
    }

    #[pyo3(signature = (**kwds))]
    pub fn compile(&mut self, kwds: Option<&PyDict>) {
        println!("Model::compile");

        match kwds {
            Some(args) => {
                println!("\r\n{:?}", args);
            }
            _ => {
                panic!("No arguments for compile!");
            }
        }

        self.graph.compile(kwds);
    }

    pub fn forward(&self) {
        println!("Model::forward");
        self.graph.forward();
    }

    pub fn backward(&self) {
        println!("Model::backward");
        self.graph.backward();
    }

    pub fn update(&self) {
        println!("Model::update");
        self.graph.update();
    }

    pub fn zero_gradients(&self) {
        println!("Model::zero_gradients");
        self.graph.zero_gradients();
    }

    #[pyo3(text_signature = "($self, pyop)")]
    pub fn add_operator(&mut self, pyop: &mut PyOperator) {
        // let _optype = OpType::try_from(optype as u32);
        let idx = self.num_of_operators();
        // pyop.id = id;
        print!("Op: {}, ", pyop.op_type.as_str());
        for (key, value) in &pyop.params {
            print!("{}:{}, ", key, value)
        }
        print!("\n");
        // let mut params_ : args;
        // params_.extend(args.into_iter());
        let operator = Box::new(Operator::new(pyop.op_type, pyop.params.clone()));

        let handle_ptr: *const Operator = &*operator;

        self.graph.operators.push(operator);

        // let ptr = &mut self.graph.operators[idx];
        pyop.raw_ptr = handle_ptr as u64;
        // return Python::with_gil(|py| -> PyResult<Py<PyOperator>> {
        //     let foo: Py<PyOperator> = Py::new(py, PyOperator {id: id, op_type : optype})?;
        //     return Ok(foo)
        // });

        //

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

        println!(
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
            op_type: op_type,
            params: HashMap::<String, String>::new(),
            raw_ptr: 0,
        };

        match args {
            Some(para) => {
                for key in para.keys() {
                    if key.to_string() != "input" {
                        op.params
                            .insert(key.to_string(), para.get_item(key).unwrap().to_string());
                    }
                }

                self.add_operator(&mut op);

                for key in para.keys() {
                    if key.to_string() == "input" {
                        let ret = para.get_item(key).unwrap().extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();
                        match ret {
                            Ok(v) => match op.add_input(&v) {
                                Ok(_) => {
                                    op.calculate_output();
                                }
                                _ => {}
                            },
                            _ => {
                                panic! {"Not a valid input type!"};
                            }
                        }
                    } else if op_type == OpType::CONCAT && key.to_string() == "tensors" {
                        let ret = para
                            .get_item(key)
                            .unwrap()
                            .extract::<Vec<PyRef<TensorF32>>>(); // .downcast::<TensorF32>();
                        match ret {
                            Ok(vlist) => {
                                for v in vlist {
                                    match op.add_input(&v) {
                                        Ok(_) => {
                                            println!(
                                                "A list of input tensors added for operator {:?}!",
                                                op_type
                                            );
                                        }
                                        _ => {}
                                    }
                                }
                                op.calculate_output();
                            }
                            _ => {
                                panic! {"Not a valid input type!"};
                            }
                        }
                    }
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
    pub fn input(&self) {}
    pub fn output(&self) {}

    fn handle_operator(&mut self, op_type: OpType, kwds: Option<&PyDict>) -> Py<PyOperator> {
        match kwds {
            Some(args) => {
                println!("\r\n{:?}", args);

                return self.add_layer(op_type, kwds);
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
    pub fn pool2d(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::POOL2D, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn batch_norm(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::BATCH_NORM, kwds)
    }

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

    pub fn multihead_attention(&self) {}

    pub fn layer_norm(&self) {}

    pub fn embedding(&self) {}

    pub fn expand(&self) {}

    #[pyo3(signature = (**kwds))]
    pub fn softmax(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SOFTMAX, kwds)
    }

    pub fn transpose(&self) {}

    pub fn reshape(&self) {}

    pub fn dropout(&self) {}

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
    pub fn concat(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::CONCAT, kwds)
    }

    #[pyo3(signature = (**kwds))]
    pub fn split(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::SPLIT, kwds)
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

    pub fn identity(&self) {}

    pub fn attribute(&self) {}

    pub fn getitem(&self) {}

    pub fn getattr(&self) {}

    pub fn init_param(&self) {}

    pub fn float(&self) {}
}
