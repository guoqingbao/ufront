use num::Signed;
use num;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Buffer<T: Clone + Signed> {
    length : usize,
    raw_values : Vec<T>,
}

impl<T: Clone + Signed> Buffer<T> {
    pub fn new(length : usize, values : Option<Vec<T>>) -> Buffer<T> {
        match values {
            Some(v) => {
                Buffer { length: length, raw_values : v}
            }
            _ => {
                //lazy data buffer
                Buffer { length: length, raw_values : Vec::<T>::new()}
            }
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        if self.raw_values.len() == 0 && self.length > 0 {
            return vec![num::zero(); self.length]
        }
        else {
            self.raw_values.to_vec()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataBuffer<T: Clone + Signed> {
    CPUDataBuffer(Buffer<T>),
    GPUDataBuffer(Buffer<T>),
    GCUDataBuffer(Buffer<T>),
}
