#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataBuffer<T> {
    CPUDataBuffer(Vec<T>),
    GPUDataBuffer(Vec<T>),
    GCUDataBuffer(Vec<T>),
}
