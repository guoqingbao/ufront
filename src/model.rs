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
use crate::graph::Graph;
use crate::graph::GraphTrait;
use crate::operator::Operator;
use crate::operator::PyOperator;
use crate::prelude::Tensor;
use crate::tensor::TensorF32;
use crate::types::DataType;
use crate::types::OpType;
use indexmap::IndexMap;

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
}

#[pymethods]
impl Model {
    #[new]
    pub fn new() -> Model {
        println!("Model::new");
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
        println!("Model::compile");

        match kwds {
            Some(args) => {
                println!("\r\n{args:?}");
            }
            _ => {
                panic!("No arguments for compile!");
            }
        }

        self.graph.compile(kwds);
        self.build_ssa_ids();
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

        print!("Op: {}, ", pyop.op_type.as_str());
        let mut idxmap = IndexMap::<String, String>::new();
        for (key, value) in &pyop.params {
            print!("{key}:{value}, ");
            idxmap.insert(key.to_string(), value.to_string());
        }
        println!();

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

    pub fn dump_ir(&self) -> String {
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

        // println!("{:?}", self.args);
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
        let mut argstr = "".to_string();
        match args {
            Some(para) => {
                for key in para.keys() {
                    if key.to_string() != "input" {
                        op.params
                            .insert(key.to_string(), para.get_item(key).unwrap().to_string());
                    } else {
                        argstr = para.get_item(key).unwrap().to_string();
                    }
                }

                self.add_operator(&mut op);

                if op_type == OpType::CONCAT {
                    if para.contains("tensors").is_ok() {
                        let ret = para
                            .get_item("tensors")
                            .unwrap()
                            .extract::<Vec<PyRef<TensorF32>>>(); // .downcast::<TensorF32>();
                        match ret {
                            Ok(vlist) => {
                                for v in vlist {
                                    if op.add_input(&v).is_ok() {
                                        println!(
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
                } else if op_type == OpType::ADD || op_type == OpType::MULTIPLY || 
                        op_type == OpType::BATCH_MATMUL || op_type == OpType::LESS ||
                        ((op_type == OpType::SLICE) && para.contains("input").is_err() )    {
                    if para.contains("x").is_ok() && para.contains("y").is_ok() {
                        let x = para.get_item("x").unwrap().extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();
                        let y = para.get_item("y").unwrap().extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();
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
                    if para.contains("q").is_ok()
                        && para.contains("k").is_ok()
                        && para.contains("v").is_ok()
                    {
                        let q = para.get_item("q").unwrap().extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();
                        let k = para.get_item("k").unwrap().extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();
                        let v = para.get_item("v").unwrap().extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();

                        match (q, k, v) {
                            (Ok(q1), Ok(k1), Ok(v1)) => {
                                if op.add_input(&q1).is_ok()
                                    && op.add_input(&k1).is_ok()
                                    && op.add_input(&v1).is_ok()
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
                } else if op_type == OpType::PARAMETER || op_type == OpType::TENSOR {
                    if para.contains("np_tensor").is_ok() && para.contains("dtype").is_ok() {
                        let np_tensor = para
                            .get_item("np_tensor")
                            .unwrap()
                            .extract::<PyReadonlyArrayDyn<f32>>();
                        let dtype = para
                            .get_item("dtype")
                            .unwrap()
                            .extract::<DataType>()
                            .unwrap();
                        match dtype {
                            DataType::Float
                            | DataType::Double
                            | DataType::Int32
                            | DataType::Int64 => {}
                            _ => {
                                panic! {"Not supported type!"};
                            }
                        }

                        match np_tensor {
                            Ok(_np_tensor) => {
                                let mut tensor = TensorF32 {
                                    tensor: None,
                                    name: para.get_item("name").unwrap().to_string(),
                                    dtype: dtype,
                                };
                                tensor.set_ndarray(_np_tensor);
                                if op.add_input(&tensor).is_ok() {
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
                } else if op_type == OpType::MASKEDFILL {
                    if para.contains("input").is_ok()
                        && para.contains("mask").is_ok()
                        && para.contains("value").is_ok()
                    {
                        let input = para.get_item("input").unwrap().extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();
                        let mask = para.get_item("mask").unwrap().extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();

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
                } else if para.contains("input").is_ok() {
                    let ret = para
                        .get_item("input")
                        .unwrap()
                        .extract::<PyRef<TensorF32>>(); // .downcast::<TensorF32>();
                    match ret {
                        Ok(v) => {
                            if v.name.find("input") == Some(0) {
                                self.args.insert(v.name.clone(), argstr.clone());
                                self.argshapes.insert(v.name.clone(), v.get_ir());
                            }
                            if op.add_input(&v).is_ok() {
                                op.calculate_output();
                            }
                        }
                        _ => {
                            panic! {"Not a valid input type!"};
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

    pub fn create_tensor(
        &mut self,
        shape: Vec<usize>,
        dtype: DataType,
        requires_grad: bool,
        name: String,
    ) -> Py<TensorF32> {
        // let sp = shape.as_slice().to_vec();
        match dtype {
            DataType::Float | DataType::Double | DataType::Int32 | DataType::Int64 | DataType::Bool => {
                let shape = shape.to_vec();
                let tensor = Some(Tensor::<f32> {
                    shape: shape.clone(),
                    data_buffer: DataBuffer::CPUDataBuffer(vec![0f32; shape.iter().product()]),
                });
                println!("Tensor initialized with {shape:?} dimension within Rust");
                Python::with_gil(|py| {
                    Py::new(
                        py,
                        TensorF32 {
                            tensor,
                            name: name.clone(),
                            dtype: dtype,
                        },
                    )
                    .unwrap()
                })
            }
            _ => {
                panic!("Not supported type at the moment!");
            }
        }
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
        println!("{:?}\r\n", kwds);
        let op_type = OpType::CALL;
        let mut op = PyOperator {
            op_type,
            params: HashMap::<String, String>::new(),
            raw_ptr: 0,
        };
        let mut argstr = "".to_string();
        match kwds {
            Some(para) => {
                for key in para.keys() {
                    if key.to_string() != "input" {
                        op.params
                            .insert(key.to_string(), para.get_item(key).unwrap().to_string());
                    } else {
                        argstr = para.get_item(key).unwrap().to_string();
                    }
                }

                let callback = para.get_item("callback").unwrap().extract::<PyObject>();
                let func_args = para.get_item("args").unwrap().extract::<&PyDict>(); // .downcast::<TensorF32>();
                self.add_operator(&mut op);

                match (callback, func_args) {
                    (Ok(_callback), Ok(_func_args)) => {
                        println!("Function args  {:?}", _func_args);

                        for key in _func_args.keys() {
                            let x = _func_args
                                .get_item(key)
                                .unwrap()
                                .extract::<PyRef<TensorF32>>();
                            match x {
                                Ok(_x) => {
                                    if op.add_input(&_x).is_ok() {
                                        println!(
                                            "A list of input tensors added for operator {op_type:?}!"
                                        );
                                    }
                                }
                                _ => {
                                    println!("{:?} is not tensor argument!", key);
                                }
                            }
                        }

                        match _callback.call(py, (), kwds) {
                            Ok(ret) => {
                                // if para.contains("return").is_ok() {
                                println!("Model::add_layer calculate output for {:?} by calling the external function {}", 
                                            op_type, para.get_item("func").unwrap().to_string());
                                let tensor = ret.extract::<PyRef<TensorF32>>(py);

                                match tensor {
                                    Ok(v) => {
                                        op.add_output(&v);
                                        println!("Output tensor with shape {:?} created within Rust for operator {:?}!", v.get_shape(py), op_type);
                                    }
                                    // _ => panic!("Invalid tensor outputs!"),
                                    _ => {
                                        println!("The return value from function call {} is not tensor output!", para.get_item("func").unwrap().to_string());
                                    }
                                }
                            }
                            Err(e) => {
                                println!("No return values of the callback! {:?}", e);
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

    pub fn input(&mut self) {}
    pub fn output(&self) {}

    #[pyo3(signature = (**kwds))]
    pub fn eq(&mut self, kwds: Option<&PyDict>) -> Py<PyOperator> {
        self.handle_operator(OpType::EQ, kwds)
    }

    fn handle_operator(&mut self, op_type: OpType, kwds: Option<&PyDict>) -> Py<PyOperator> {
        match kwds {
            Some(args) => {
                println!("\r\n{args:?}");
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

    pub fn embedding(&self) {}

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

    pub fn float(&self) {}

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
}
