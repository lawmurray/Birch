# NumBirch C++ Library

NumBirch is a C++ library for asynchronous numerics on CPU and GPU. Features
include:

* **Numerical functions:** Batched standard and special math functions, linear
  algebra.
* **Multithread and multistream computation:** Array class with copy-on-write
  and element-as-future semantics, multithread host and multistream device.
* **Multistream memory management:** Unified memory across host and device
  with arenas, pooling, and recycling.

NumBirch supports two different backends based on the following libraries, as well as custom code:

| Backend | Device | Memory            | Linear algebra   | Special math functions | Primitives |
| ------- | ------ | ----------------- | ---------------- | ---------------------- | ---------- |
| Eigen   | CPU    | host, jemalloc    | Eigen            | Eigen                  |            |
| CUDA    | GPU    | unified, jemalloc | cuBLAS, cuSOLVER | Eigen                  | CUB        |
| oneAPI  | GPU    | unified, jemalloc | MKL              | Eigen                  | DPL        |

!!! attention
    The oneAPI backend that is neither complete nor working at the moment.

## License

NumBirch is open source software. It is licensed under the Apache License,
Version 2.0 (the "License"); you may not use it except in compliance with the
License. You may obtain a copy of the License at
<http://www.apache.org/licenses/LICENSE-2.0>.

## Installation

### From source: Eigen backend (default)

Requirements are:

  * GNU autoconf, automake, and libtool
  * [Eigen](https://eigen.tuxfamily.org)

Proceed as follows:

```
./bootstrap
./configure
make
make install
```

If installing system wide, the last command may require `sudo`.

### From source: CUDA backend

To use the CUDA backend you will need an Nvidia GPU. You are unlikely to see
speedups over the Eigen backend unless your profile is dominated by large
matrix operations (of at least hundreds of rows and columns).

In addition to the requirements for the default backend, the CUDA backend
requires [CUDA](https://developer.nvidia.com/cuda-downloads) and a custom
build of [jemalloc](http://jemalloc.net/), instructions for which are below.

#### 1. Build jemalloc

Download the latest [jemalloc source](https://github.com/jemalloc/jemalloc)
and extract it. From within its main directory run:
```
./configure \
    --disable-shared \
    --enable-static \
    --disable-stats \
    --disable-initial-exec-tls \
    --disable-doc \
    --with-jemalloc-prefix=numbirch_ \
    --with-install-suffix=_numbirch
make
make install
```
Replace `./configure` with `./autogen.sh` if the former does not exist. If
installing system wide, the last command may require `sudo`.

The options `--with-jemalloc-prefix=numbirch_` and
`--with-install-suffix=_numbirch` are critical and no other values should be
used. NumBirch is hard-coded to work with these values.

There is a script `install_cuda_prereq` in the main directory that is used
when packaging NumBirch. Its contents may be a helpful guide here if you get
stuck.

#### 2. Install the `nvcc_wrapper` script

The `nvcc_wrapper` script is a thin wrapper around `nvcc` that resolves some
incompatibilites between it and the GNU Autotools, specifically around
command-line options that lead to incorrect dependency detection and shared
library behavior. It must be installed somewhere on your `$PATH` but *outside*
the NumBirch source directory to work correctly. It is, however, only required
for building NumBirch; developers who expect to rebuild NumBirch frequently
may like to install it to a permanent location, users who expect to rebuild
infrequently may like to install to a temporary location instead:

    mkdir $HOME/tmpbin
    export OLDPATH=$PATH
    export PATH=$HOME/tmpbin:$OLDPATH
    cp nvcc_wrapper $HOME/tmpbin/.

#### 3. Install NumBirch

Finally, in the root directory of the NumBirch sources, build NumBirch with
the CUDA backend:
```
./bootstrap
./configure --enable-cuda
make
make install
```

If installing system wide, the last command may require `sudo`.

#### 4. Clean up

If you installed `nvcc_wrapper` to a temporary location, you may now clean up:

    rm $HOME/tmpbin/nvcc_wrapper
    rmdir $HOME/tmpbin
    export PATH=$OLDPATH

### From source: oneAPI backend

!!! attention
    The oneAPI backend is neither complete nor working at the moment.

The oneAPI requires the [Intel oneAPI Base
Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.g8tsv3).
Once installed, run

```
. /opt/intel/oneapi/setvars.sh
```

to set up your environment to use it.

Then, to build with the oneAPI backend:

```
./bootstrap
./configure --enable-oneapi
make
make install
```

To run using the oneAPI backend you will need an Intel GPU, such as the
integrated graphics on an Intel CPU.

## Getting started

To use NumBirch in your own code, include the header file:
```
#include <numbirch.hpp>
```
By default, this sets the type `real` to `double`. To instead set it to
`float`, use:
```
#define NUMBIRCH_REAL float
#include <numbirch.hpp>
```
or define `-DNUMBIRCH_REAL=float` when compiling. Also add (at least)
`-std=c++17` to the compile line, and link the appropriate library:

- `numbirch` for the Eigen backend in double precision,
- `numbirch-single` for the Eigen backend in single precision,
- `numbirch-cuda` for the CUDA backend in double precision,
- `numbirch-cuda-single` for the CUDA backend in single precision.

A simple "Hello world" program is as follows. In `hello.cpp`:

```
#include <numbirch.hpp>
#include <iostream>

int main() {
  Matrix<real> A = {{1.0, 0.5}, {0.0, 1.0}};
  Vector<real> x = {10.0, 2.0};
  auto y = A*x;
  std::cerr << "Hello " << y << "!" << std::endl;
  return 0;
}
```
Compile and link:
```
g++ -std=c++17 -lnumbirch -o hello hello.cpp
```
and run:
```
./hello
```

## Interface

NumBirch supports four different types for all operations: `bool`, `int`, and
`real`. The latter is either `float` or `double` according to the
configuration. Any function or operator may be called with arguments of these
types, or arguments that are arrays of these types. While the NumBirch
interface is made up of template functions, these templates are explicitly
instantiated for all valid types in the NumBirch library linked when building.
This design allows the use of a regular C++ toolchain for compiling client
projects, rather than the more complex toolchains required for some of the
backends. Much simpler.

While the interface of NumBirch may look a little complicated with its
template code, it works with few surprises. You can declare matrices and
vectors:
```
Matrix<real> A;
Vector<real> x;
```
and e.g. multiply them together:
```
auto y = A*x;
```
to get, in this case, `y` of type `Vector<real>`. There are numerous
*transformations* for standard math functions that you can batch across
arrays, e.g.:
```
auto z = sin(y);
```
to get, again, `z` of type `Vector<real>`. When scalars are required you can
use primitive scalar types:
```
auto a = 2.0;
z = 2.0*z;
```
In fact, `Matrix<T>` is just an alias for `Array<T,2>`, `Vector<T>` for
`Array<T,1>`, and `Scalar<T>` for `Array<T,0>`. The latter can work like a
primitive scalar:
```
Scalar<real> b = 2.0;
z = b*z;
```
but has a role in multistreaming computing, described later.

### Understanding type traits

Most functions are defined with generic parameter types. The C++ idiom of
*SFINAE* ("Substitution Failure Is Not An Error") is used to restrict the
accepted types according to *type traits*. Consider, for example, the addition
operator. Its full signature is:

```
template<class T, class U, class = std::enable_if_t<is_numeric_v<T> &&
    is_numeric_v<U>,int>>
implicit_t<T,U> operator+(const T& x, const U& y);
```

The parameter types are generic: `T` and `U`, but according to the SFINAE in
the template parameter list, they must be *numeric*. The type traits that you
may encounter are:

| Name             | Trait                 | Description                                                  |
| ---------------- | :-------------------- | :----------------------------------------------------------- |
| *integral*       | `is_integral_v`       | `bool` or `int`, c.f. [`std::is_integral`](https://en.cppreference.com/w/cpp/types/is_integral) |
| *floating point* | `is_floating_point_v` | `float` or `double`, c.f. [`std::is_floating_point`](https://en.cppreference.com/w/cpp/types/is_integral) |
| *arithmetic*     | `is_arithmetic_v`     | `bool`, `int`, `float`, or `double`, c.f. [`std::is_arithmetic`](https://en.cppreference.com/w/cpp/types/is_integral) |
| *array*          | `is_array_v`          | `Array<T,D>` where `T` is any *arithmetic* type              |
| *numeric*        | `is_numeric_v`        | Any *arithmetic* or *array* type                             |
| *compatible*     | `is_compatible_v`     | The types must have the same number of dimensions. Zero-dimensional arrays (e.g. `Array<double,0>` a.k.a. `Scalar<double>`) are also compatible with scalars (e.g. `double`). |

### Implicit conversion

Arithmetic types promote in the order `bool` to `int` to `real`. The return
type of an operation between two or more different types will be the highest
type in this promotion order, unless explicitly specified otherwise; e.g. an
operation between a `bool` and `int` yields an `int`, between an `int` and
`real` yields a `real`.

The same extends to arrays; e.g. an operation between an `Array<int,D>` and
`Array<real,D>` yields an `Array<real,D>`.

These numerical promotion rules differ from the broader rules of C++.
Specifically, under C++ rules, an operation between an `int` and a `float`
promotes to a `double`, not to a `float` as here (if `real` is defined as
`float`). The choice to deviate from the C++ rules reflects that working in
single precision has a significant performance advantage on much modern
hardware, such as GPUs, where the FLOPS ratio between single- and
double-precision can be as high as 32:1. Under the C++ rules, it can be a
little too easy to promote from single- to double-precision operations
unintentionally. The NumBirch rules eliminate some common situations where
this occurs.

## Multistreaming computation

NumBirch uses a multistreaming computing model. Asynchronicity is between the
*host*, on which your code is running, and the *device* on which NumBirch
kernels are launched. The host is the CPU. If using the default backend, the
device is also the CPU; in this case computation is synchronous, so that a
call to a NumBirch function will not return until its computation is complete.
If using e.g. the CUDA backend, the device is a GPU; in this case computation
is asynchronous, so that a call to a NumBirch function may return before its
computation is complete. In this latter case understanding the asynchronous
computing model may be necessary to obtain optimal performance.

The general principal is that *arrays are element-wise futures*. An array
returned from one function can be passed as an argument to another function
with no synchronization required---the contents of the array may not have been
computed yet, but you don't need it yet either, and the backend will stream
these function calls one after another when the device is ready. Arrays can
even go out of scope and be destroyed without synchronization---memory
allocation and deallocation is streamed asynchronously by the backend too, so
a deallocation of an array will not occur until all computation involving it
has completed.

Synchronization is only required when accessing an array *element-wise*. This
is handled internally by the numbirch::Array class, so there is nothing extra
for you to do in your own code except be aware that this has performance
implications: your code will block until all computation on the device for the
current host thread is complete. Element-wise access includes accessing
individual elements (called *dicing*, as opposed to *slicing* in the Array
idioms), or obtaining an iterator over elements. This idiom avoids
synchronization until you are ready to access the result. See numbirch::Array
for further details.

In this way, arrays act as *futures* (c.f.
[`std::future`](https://en.cppreference.com/w/cpp/thread/future)), providing
an easy mechanism to synchronize automatically when necessary. Some functions
even return an `Array<T,0>` a.k.a `Scalar<T>` or `Future<T>`. Such a
zero-dimensional array has just one element, but works like any other array
with respect to being an element-wise future on that element. Typical
functions that work this way are those that perform reductions, e.g. sum(),
trace(), ldet().

## Multistream memory management

### Copy on write

All arrays are copy-on-write. This allows multiple arrays to share the same
underlying memory for as long as they are only read, with a new buffer
allocated and the contents copied over when a write is attempted. There is a
small overhead in the use of atomic spin locks when this occurs, but for many
use cases, a significant saving in the elimination of unnecessary copies.

### Memory pooling

NumBirch uses [jemalloc](http://jemalloc.net/) with custom extent hooks to
allocate unified memory, i.e. memory that can be accessed on both host and
device. Separate arenas are used for memory that is used only on the device
(e.g. for temporaries within a single numeric function) versus that which may
be used on both (e.g. for the buffers used by `numbirch::Array`). In all
cases, however, unified memory is used, as jemalloc itself may need to access
the memory on host. Each arena has its own thread-local memory pool. Memory
allocations (e.g. via `cudaMallocManaged()`) occur either to extend the size
of the extents used by pools, or for very large allocations. These tend to
occur early on in program execution and their frequency diminish in time,
although this will depend on the memory profile of the particular program.

# Q&A

### Does NumBirch support mixed-precision floating point operations?

No.

Firstly, backends often do not support mixed precision for certain operations
(consider e.g. BLAS and LAPACK interfaces). Secondly, the need to support
forward (evaluation) and backward (gradient) computations for reverse-mode
automatic differentiation means that floating point promotion occurs in both
directions, causing incompatibilities with those backends one way or the
other. The choice was made to avoid this complication, at least for now.
