use std::any::{Any, TypeId};
use std::mem::size_of;
use half::f16;
use half::bf16;

pub trait NumType {}
impl NumType for f64 {}
impl NumType for f32 {}
impl NumType for f16 {}
impl NumType for bf16 {}
impl NumType for i32 {}
impl NumType for i64 {}
impl NumType for bool {}

pub fn equals<U: 'static, V: 'static>() -> bool {
    TypeId::of::<U>() == TypeId::of::<V>() && size_of::<U>() == size_of::<V>()
}
pub fn cast_ref<U: 'static, V: 'static>(u: &U) -> Option<&V> {
    if equals::<U, V>() {
        Some(unsafe { std::mem::transmute::<&U, &V>(u) })
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
enum RawBuffer {
    FP64Buffer(Vec<f64>),
    FP32Buffer(Vec<f32>),
    FP16Buffer(Vec<f16>),
    BF16Buffer(Vec<bf16>),
    I32Buffer(Vec<i32>),
    I64Buffer(Vec<i64>),
    BoolBuffer(Vec<bool>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Buffer {
    length : usize,
    raw_values : RawBuffer,
}

impl Buffer {
    pub fn new<T: NumType + 'static>(length : usize, values : Option<Vec<T>>) -> Buffer {
            match values {
                Some(v) => {
                    if TypeId::of::<T>() == TypeId::of::<f32>() {
                        if let Some(ret) = cast_ref::<Vec<T>, Vec<f32>>(&v) {
                            Buffer { length: length, raw_values : RawBuffer::FP32Buffer(ret.to_vec())}
                        } else {
                            panic!("Unabel to convert to vector of f32!")
                        }
                    } else if TypeId::of::<T>() == TypeId::of::<f16>() {
                        if let Some(ret) = cast_ref::<Vec<T>, Vec<f16>>(&v) {
                            Buffer { length: length, raw_values : RawBuffer::FP16Buffer(ret.to_vec())}
                        } else {
                            panic!("Unabel to convert to vector of f16!")
                        }
                    } else if TypeId::of::<T>() == TypeId::of::<bf16>() {
                        if let Some(ret) = cast_ref::<Vec<T>, Vec<bf16>>(&v) {
                            Buffer { length: length, raw_values : RawBuffer::BF16Buffer(ret.to_vec())}
                        } else {
                            panic!("Unabel to convert to vector of bf16!")
                        }
                    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                        if let Some(ret) = cast_ref::<Vec<T>, Vec<f64>>(&v) {
                            Buffer { length: length, raw_values : RawBuffer::FP64Buffer(ret.to_vec())}
                        } else {
                            panic!("Unabel to convert to vector of fp64!")
                        }
                    } else if TypeId::of::<T>() == TypeId::of::<i32>() {
                        if let Some(ret) = cast_ref::<Vec<T>, Vec<i32>>(&v) {
                            Buffer { length: length, raw_values : RawBuffer::I32Buffer(ret.to_vec())}
                        } else {
                            panic!("Unabel to convert to vector of i32!")
                        }
                    } else if TypeId::of::<T>() == TypeId::of::<i64>() {
                        if let Some(ret) = cast_ref::<Vec<T>, Vec<i64>>(&v) {
                            Buffer { length: length, raw_values : RawBuffer::I64Buffer(ret.to_vec())}
                        } else {
                            panic!("Unabel to convert to vector of i64!")
                        }
                    } else if TypeId::of::<T>() == TypeId::of::<bool>() {
                        if let Some(ret) = cast_ref::<Vec<T>, Vec<bool>>(&v) {
                            Buffer { length: length, raw_values : RawBuffer::BoolBuffer(ret.to_vec())}
                        } else {
                            panic!("Unabel to convert to vector of bool!")
                        }
                    } else {
                        panic!("Not supported data type!")
                    }
                }
                _ => {
                    //lazy data buffer
                    if TypeId::of::<T>() == TypeId::of::<f32>() {
                        Buffer { length: length, raw_values : RawBuffer::FP32Buffer(Vec::<f32>::new())}
                    } else if TypeId::of::<T>() == TypeId::of::<f16>() {
                        Buffer { length: length, raw_values : RawBuffer::FP16Buffer(Vec::<f16>::new())}
                    } else if TypeId::of::<T>() == TypeId::of::<bf16>() {
                        Buffer { length: length, raw_values : RawBuffer::BF16Buffer(Vec::<bf16>::new())}
                    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                        Buffer { length: length, raw_values : RawBuffer::FP64Buffer(Vec::<f64>::new())}
                    } else if TypeId::of::<T>() == TypeId::of::<i32>() {
                        Buffer { length: length, raw_values : RawBuffer::I32Buffer(Vec::<i32>::new())}
                    } else if TypeId::of::<T>() == TypeId::of::<i64>() {
                        Buffer { length: length, raw_values : RawBuffer::I64Buffer(Vec::<i64>::new())}
                    } else if TypeId::of::<T>() == TypeId::of::<bool>() {
                        Buffer { length: length, raw_values : RawBuffer::BoolBuffer(Vec::<bool>::new())}
                    } else {
                        panic!("Not supported data type!")
                    }
                }
            }


    }

    pub fn to_vec<T: NumType + Copy+ 'static>(&mut self) -> Vec<T> {
        match &mut self.raw_values {
            RawBuffer::FP32Buffer(v) => {
                if v.len() == 0 && self.length > 0 {
                    v.extend(vec![0f32; self.length]);
                }
                cast_ref::<Vec<f32>, Vec<T>>(&v).unwrap().to_owned()
            }
            RawBuffer::FP16Buffer(v) => {
                if v.len() == 0 && self.length > 0 {
                    v.extend(vec![half::f16::ZERO; self.length]);
                }
                cast_ref::<Vec<f16>, Vec<T>>(&v).unwrap().to_owned()
            }
            RawBuffer::BF16Buffer(v) => {
                if v.len() == 0 && self.length > 0 {
                    v.extend(vec![half::bf16::ZERO; self.length]);
                }
                cast_ref::<Vec<bf16>, Vec<T>>(&v).unwrap().to_owned()
            }
            RawBuffer::FP64Buffer(v) => {
                if v.len() == 0 && self.length > 0 {
                    v.extend(vec![0f64; self.length]);
                }
                cast_ref::<Vec<f64>, Vec<T>>(&v).unwrap().to_owned()
            }
            RawBuffer::I32Buffer(v) => {
                if v.len() == 0 && self.length > 0 {
                    v.extend(vec![0i32; self.length]);
                }
                cast_ref::<Vec<i32>, Vec<T>>(&v).unwrap().to_owned()
            }
            RawBuffer::I64Buffer(v) => {
                if v.len() == 0 && self.length > 0 {
                    v.extend(vec![0i64; self.length]);
                }
                cast_ref::<Vec<i64>, Vec<T>>(&v).unwrap().to_owned()
            }
            RawBuffer::BoolBuffer(v) => {
                if v.len() == 0 && self.length > 0 {
                    v.extend(vec![false; self.length]);
                }
                cast_ref::<Vec<bool>, Vec<T>>(&v).unwrap().to_owned()
            }
        }

    }

    pub fn as_ptr<T: Any>(&self) -> *const T {
        match &self.raw_values {
            RawBuffer::FP32Buffer(v) => {
                return v.as_ptr() as *const T;
            }
            RawBuffer::FP16Buffer(v) => {
                return v.as_ptr() as *const T;
            }
            RawBuffer::BF16Buffer(v) => {
                return v.as_ptr() as *const T;
            }
            RawBuffer::FP64Buffer(v) => {
                return v.as_ptr() as *const T;
            }
            RawBuffer::I32Buffer(v) => {
                return v.as_ptr() as *const T;
            }
            RawBuffer::I64Buffer(v) => {
                return v.as_ptr() as *const T;
            }
            RawBuffer::BoolBuffer(v) => {
                return v.as_ptr() as *const T;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataBuffer {
    CPUDataBuffer(Buffer),
    GPUDataBuffer(Buffer),
    GCUDataBuffer(Buffer),
}

impl DataBuffer {
    pub fn get_type_id(&self) -> TypeId {
        match self {
            DataBuffer::CPUDataBuffer(buf) | DataBuffer::GPUDataBuffer(buf) | DataBuffer::GCUDataBuffer(buf) => {
                // buf.raw_buffer.element_type_id()
                match &buf.raw_values {
                    RawBuffer::FP32Buffer(_) => {
                        return TypeId::of::<f32>();
                    }
                    RawBuffer::FP16Buffer(_) => {
                        return TypeId::of::<f16>();
                    }
                    RawBuffer::BF16Buffer(_) => {
                        return TypeId::of::<bf16>();
                    }
                    RawBuffer::FP64Buffer(_) => {
                        return TypeId::of::<f64>();
                    }
                    RawBuffer::I32Buffer(_) => {
                        return TypeId::of::<i32>();
                    }
                    RawBuffer::I64Buffer(_) => {
                        return TypeId::of::<i64>();
                    }
                    RawBuffer::BoolBuffer(_) => {
                        return TypeId::of::<bool>();
                    }
                }
            }
        }
    }
}
