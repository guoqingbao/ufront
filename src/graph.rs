use crate::operator::Operator;
use pyo3::types::PyDict;
pub trait GraphTrait {
    fn compile(&self, kwds: Option<&PyDict>);
    fn forward(&self);
    fn backward(&self);
    fn update(&self);
    fn zero_gradients(&self);
}

pub struct Graph {
    pub operators: Vec<Box<Operator>>,
}

impl GraphTrait for Graph {
    fn compile(&self, kwds: Option<&PyDict>) {
        println!("Graph::compile");
    }
    fn forward(&self) {
        println!("Graph::forward");
    }
    fn backward(&self) {
        println!("Graph::backward");
    }
    fn update(&self) {
        println!("Graph::update");
    }
    fn zero_gradients(&self) {
        println!("Graph::zero_gradients");
    }
}
