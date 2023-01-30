use std::collections::HashMap;

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
use types::OpType;
use operator::PyOperator;
use numpy::pyo3::Python;
use numpy::ndarray::array;
use numpy::{ToPyArray, PyArray};
use numpy::IntoPyArray;


fn main() {
    pyo3::prepare_freethreaded_python();
    let mut model = Model::new();
    let mut params = HashMap::<String, String>::new();
    params.insert("input_channel".to_string(), "1".to_string());
    params.insert("output_channel".to_string(), "16".to_string());
    params.insert("kernel_size".to_string(), "[3,3]".to_string());
    let mut operator = PyOperator { op_type : OpType::CONV2D, params : params, raw_ptr : 0};

    model.add_operator(&mut operator);
    Python::with_gil(|py| {
        let arr = array![[1.0f32, 2.0], [3.0, 4.0]].to_pyarray(py);
        // vec![1.0f32, 2.0, 3.0].into_pyarray(py)
        operator.add_input_ndarray(arr.to_dyn().readonly());
    });

    
    println!("{}", operator.raw_ptr);
    println!("{}", operator.num_of_inputs());

    println!("{}", operator.num_of_outputs());

    // model.compile();
    model.forward();

    // tensorf32.set_ndarray(arr)
    // print("Obtained dimension: ", tensorf32.get_dims())

    // a = tensorf32.get_ndarray()
    Python::with_gil(|py| { 
        let a = operator.get_input_ndarray(0, py);
        println!("Retrive tensor from Rust: {:?} \n ", a);
    }
    );

    model.remove_operator(&mut operator);
    // print(operator.op_type)

}