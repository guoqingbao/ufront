use crate::operator::Operator;
use pyo3::types::PyDict;
use log::info;
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
        info!("Graph::compile");
    }
    fn forward(&self) {
        info!("Graph::forward");
    }
    fn backward(&self) {
        info!("Graph::backward");
    }
    fn update(&self) {
        info!("Graph::update");
    }
    fn zero_gradients(&self) {
        info!("Graph::zero_gradients");
    }
}
