use std::any::{Any, TypeId};
use std::clone;
use std::mem::size_of;
use num::Signed;
use num;
// use data_buffer::DataBuffer as RawBuffer;
use half::f16;
use half::bf16;

pub trait NumType {}

impl NumType for f32 {}
impl NumType for f16 {}
impl NumType for bf16 {}

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
    FP32Buffer(Vec<f32>),
    FP16Buffer(Vec<f16>),
    BP16Buffer(Vec<bf16>),
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
                    } else {
                        if let Some(ret) = cast_ref::<Vec<T>, Vec<bf16>>(&v) {
                            Buffer { length: length, raw_values : RawBuffer::BP16Buffer(ret.to_vec())}
                        } else {
                            panic!("Unabel to convert to vector of bf16!")
                        }
                    }
                }
                _ => {
                    //lazy data buffer
                    if TypeId::of::<T>() == TypeId::of::<f32>() {
                        Buffer { length: length, raw_values : RawBuffer::FP32Buffer(Vec::<f32>::new())}
                    } else if TypeId::of::<T>() == TypeId::of::<f16>() {
                        Buffer { length: length, raw_values : RawBuffer::FP16Buffer(Vec::<f16>::new())}
                    } else {
                        Buffer { length: length, raw_values : RawBuffer::BP16Buffer(Vec::<bf16>::new())}
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
            RawBuffer::BP16Buffer(v) => {
                if v.len() == 0 && self.length > 0 {
                    v.extend(vec![half::bf16::ZERO; self.length]);
                }
                cast_ref::<Vec<bf16>, Vec<T>>(&v).unwrap().to_owned()
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
            RawBuffer::BP16Buffer(v) => {
                return v.as_ptr() as *const T;
            }
        }
    }
}


// #[derive(Debug, Clone, PartialEq, Hash)]
// pub struct Buffer {
//     length : usize,
//     // raw_values : Vec<T>,
//     raw_buffer : RawBuffer,
// }

// impl Buffer {
//     pub fn new<T: Any>(length : usize, values : Option<Vec<T>>) -> Buffer {
//         match values {
//             Some(v) => {
//                 Buffer { length: length, raw_buffer : RawBuffer::from_vec(v)}
//             }
//             _ => {
//                 //lazy data buffer
//                 Buffer { length: length, raw_buffer : RawBuffer::with_capacity::<T>(length)}
//             }
//         }
//     }

//     pub fn to_vec<T: NumType + Copy+ 'static>(&mut self) -> Vec<T> {
//         if self.raw_buffer.is_empty() && self.length > 0 {
//             if TypeId::of::<T>() == TypeId::of::<f16>() {
//                 self.raw_buffer.fill(half::f16::ZERO);
//             } else {
//                 self.raw_buffer.fill(num::zero::<f32>());
//             }

//         }

//         return self.raw_buffer.copy_into_vec().unwrap();
//     }

//     pub fn as_ptr<T: Any>(&self) -> *const T {
//         unsafe {
//             self.raw_buffer.get_unchecked(0)
//         }
//     }
// }

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
                    RawBuffer::BP16Buffer(_) => {
                        return TypeId::of::<bf16>();
                    }
                }
            }
        }
    }
}
