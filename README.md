# ParaS Compiler

ParaS (pronounced **paa-ruhs**) is an implementation of the **SYCL 2020** specification, developed to enable architecture and device agnostic unified programming model. It follows the principle of **"Code Once, Execute on All"**. The current release provides seamless execution on **CPU platforms**.

---

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Building ParaS](#building-paras)
- [Usage](#usage)
- [Environment Setup](#environment-setup)
- [Reporting Issues](#reporting-issues)

---

## Introduction

ParaS is designed to provide a portable SYCL programming environment that allows developers to write code once and execute it across different hardware architectures. The current version supports CPU execution, with future support planned for additional backends.

---

## Requirements

| Software | Version |
|----------|----------|
| Operating System | Linux |
| Compiler | Clang/LLVM v21.1.0 |
| Build System | CMake v3.28 or later |
| Build Generator | Unix Makefiles or Ninja |

> **Note**
>
> The current version of ParaS supports **CPU devices only**.

---

## Building ParaS

### 1. Clone the Repository

```bash
git clone https://github.com/ParaS-Ecosystem/ParaS-Compiler.git
```

### 2. Create a Build Directory

```bash
cd ParaS-Compiler
mkdir build
cd build
```

### 3. Configure and Build

#### Using Unix Makefiles

```bash
cmake -G "Unix Makefiles" \
    ../ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DPARAS_BASE_COMPILER=clang++ \
    -DPARAS_INSTALL_PREFIX=/path/to/install/dir \
    -DCMAKE_CXX_FLAGS="-std=c++17"

make -j$(nproc)
make install
```

---

## Usage

To compile a SYCL source file (for example, `sample1.cpp`):

```bash
parascc sample1.cpp <compiler_flags>
```

> **Note**
>
> `parascc` accepts all compiler flags supported by **Clang/LLVM v21.1.0**.

---

## Environment Setup

After installation, update your environment variables:

```bash
export PATH=/path/to/install/dir/bin:$PATH
export LD_LIBRARY_PATH=/path/to/install/dir/lib:$LD_LIBRARY_PATH
```

You may add these commands to your `~/.bashrc` or `~/.zshrc` to make the changes persistent.

---

## Reporting Issues

If you encounter any issues or would like to request a feature, you can:

- Open an issue in the GitHub repository under the **Issues** section.

---

## License

Please refer to the project's license file for licensing information.
