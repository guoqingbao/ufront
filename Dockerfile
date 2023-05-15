FROM ubuntu:20.04

#install LLVM/MLIR
RUN apt update && apt install -y wget cmake ninja-build gnupg

RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add

RUN apt update && apt install -y clang-16 lldb-16 lld-16 libmlir-16-dev mlir-16-tools

#openmp
RUN apt install curl libomp-16-dev -y

#pip3
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN apt install python3-distutils -y
RUN python3.8 get-pip.py

#rust & maturin
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN pip install maturin==0.15.1 
RUN pip install maturin[patchelf] 

#build subproject UFront2TOSA
COPY . /workdir/ufront
WORKDIR /workdir/ufront/cpp/UFront2TOSA/build
RUN rm -r /workdir/ufront/cpp/UFront2TOSA/build
RUN apt install zlib1g zlib1g-dev -y

RUN cmake .. -G Ninja -DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir && \
    ninja && \
    ln -s $PWD/bin/ufront-opt /usr/local/bin

WORKDIR /workdir/ufront

#build main project
ENV PATH="/root/.cargo/bin:$PATH"
# RUN maturin develop
RUN maturin build --release -i python3.8

#install ufront library that built before
RUN pip install target/wheels/ufront-0.1.1-cp38-cp38-manylinux_2_28_x86_64.whl

#run examples
RUN python3.8 examples/native_test.py