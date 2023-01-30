use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use std::fmt;
#[derive(Debug)]
pub struct RustError {
    /*
    the 'message' field that is used later on
    to be able print any message.
    */
    pub msg: &'static str,
}

// Implement the 'Error' trait for 'RustError':
impl std::error::Error for RustError {}

// Implement the 'Display' trait for 'RustError':
impl fmt::Display for RustError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Error from Rust: {}", self.msg)
    }
}

// Implement the 'From' trait for 'RustError'.
// Used to do value-to-value conversions while consuming the input value.
impl std::convert::From<RustError> for PyErr {
    fn from(err: RustError) -> PyErr {
        PyOSError::new_err(err.to_string())
    }
}
