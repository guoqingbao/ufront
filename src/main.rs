use std::collections::HashMap;

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
use model::Model;
use numpy::ndarray::array;
use numpy::pyo3::Python;
use numpy::IntoPyArray;
use numpy::{PyArray, ToPyArray};
use operator::PyOperator;
use tensor::Tensor;
use types::OpType;

fn main() {
    pyo3::prepare_freethreaded_python();
    let mut model = Model::new();
    let mut params = HashMap::<String, String>::new();
    params.insert("input_channel".to_string(), "1".to_string());
    params.insert("output_channel".to_string(), "16".to_string());
    params.insert("kernel_size".to_string(), "[3,3]".to_string());
    let mut operator = PyOperator {
        op_type: OpType::CONV2D,
        params: params.clone(),
        raw_ptr: 0,
    };

    model.add_operator(&mut operator);
    Python::with_gil(|py| {
        let arr = array![[1.0f32, 2.0], [3.0, 4.0]].to_pyarray(py);
        // vec![1.0f32, 2.0, 3.0].into_pyarray(py)
        operator.add_input_ndarray(arr.to_dyn().readonly(), "input".to_string());
    });

    let mut operator = PyOperator {
        op_type: OpType::RELU,
        params: HashMap::from([("name".to_string(), "".to_string())]),
        raw_ptr: 0,
    };

    model.add_operator(&mut operator);

    let mut operator = PyOperator {
        op_type: OpType::CONV2D,
        params: params.clone(),
        raw_ptr: 0,
    };

    model.add_operator(&mut operator);

    let mut operator = PyOperator {
        op_type: OpType::CONV2D,
        params: params.clone(),
        raw_ptr: 0,
    };

    model.add_operator(&mut operator);
    println!("{}", operator.raw_ptr);
    println!("{}", operator.num_of_inputs());

    println!("{}", operator.num_of_outputs());

    // model.compile();
    model.forward();

    // tensorf.set_ndarray(arr)
    // print("Obtained dimension: ", tensorf.get_dims())

    // a = tensorf.get_ndarray()
    Python::with_gil(|py| {
        let a = operator.get_input_ndarray(0, py);
        println!("Retrive tensor from Rust: {a:?}");
    });

    model.remove_operator(&mut operator);
    // print(operator.op_type)
}
