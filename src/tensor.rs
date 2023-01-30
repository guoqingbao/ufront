use crate::databuffer::DataBuffer;
use crate::types::DataType;
use ndarray::ArrayViewD;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
use numpy::ndarray::{Zip, ArrayD};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyArrayDyn};
use ndarray;
use ndarray::ArrayView;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

pub trait TensorTrait {
    fn get_dims(&self) -> usize;
    fn get_ndarray(&self);
    fn set_ndarray(&self);
    fn get_gradients(&self);
    fn get_owner(&self);
}

pub struct Tensor<T> {
    pub shape : Vec<usize>,
    pub data_buffer : DataBuffer<T>,
}

impl<T> TensorTrait for Tensor<T> {
    fn get_dims(&self) -> usize {
       0
    }

    fn get_ndarray(&self) {

    }

    fn set_ndarray(&self) {

    }

    fn get_gradients(&self) {

    }

    fn get_owner(&self) {

    }
}

#[pyclass]
pub struct TensorF32 {
    pub tensor : Option<Tensor<f32>>,
}

#[pymethods]
impl TensorF32 {
    #[new]
    pub fn new(x: PyReadonlyArrayDyn<f32>) -> PyResult<PyClassInitializer<Self>> {
        println!("PyTensor::new");
        let mut tensor = TensorF32 { tensor : None};
        tensor.set_ndarray(x);
        Ok(PyClassInitializer::from(tensor))
    }
    pub fn set_ndarray(&mut self,
        x: PyReadonlyArrayDyn<f32>,
    ) {
        let x = x.as_array();
        match x.as_slice() {
            Some(v) => {
                self.tensor = Some(Tensor::<f32> {
                    shape : x.shape().to_vec(), 
                    data_buffer :  DataBuffer::CPUDataBuffer(v.to_vec()),
                });
                println!("Tensor initialized with {:?} dimension within Rust", x.shape().to_vec());
            }
            _ => panic!("Invalid tensor inputs!")
        }
    }

    pub fn get_dims(&self) -> usize {
        match &self.tensor {
            Some(v) => {
                v.shape.len()
            }
            _ => panic!("Not initialized tensor!")
        }
        
    }
 
    pub fn get_ndarray<'py>(&self, py: Python<'py>) -> &'py PyArrayDyn<f32>{
        match &self.tensor {
            Some(v) => {
                match &v.data_buffer {
                    DataBuffer::CPUDataBuffer(data) => ArrayD::from_shape_vec(v.shape.to_vec(), data.to_vec()).unwrap().into_pyarray(py),
                    _ => panic!("Tensor conversion failed!")
                }
            }
            _ => panic!("Tensor not initialized!")
        }
    }
 
 
    pub fn get_gradients(&self) {
 
    }
 
    pub fn get_owner(&self) {
 
    }

    #[getter]
    pub fn get_shape(&self, py: Python<'_>,) -> Py<PyList> {
        match &self.tensor {
            Some(v) => {
                let lst = PyList::new(py, v.shape.clone());
                lst.into()
            }
            _ => panic!("Not initialized tensor!")
        }
    }
}