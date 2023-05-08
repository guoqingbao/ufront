#!/bin/bash
set -exu

bindgen \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --size_t-is-usize \
  wrapper.h -- -I../cpp/UFront2TOSA/include/CAPI \
  -I/usr/lib/gcc/x86_64-linux-gnu/9/include \
  > src/rawapi.rs