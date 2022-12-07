# NumBirch C++ Library

NumBirch is a C++ library for asynchronous numerics on CPU and GPU. Features include:

* **Numerical functions:** Batched standard and special math functions, linear algebra.
* **Asynchronous computing:** Array class with copy-on-write and element-as-future semantics.
* **Asynchronous memory management:** Unified memory across CPU and GPU with arenas and pooling.
* **Hierarchical parallelism:** Device stream per host thread for concurrent kernel execution.

<pre class="diagram">
                     |   |   |   |   |                     
                  +--'---'---'---'---'--+                  
                  |  .---------------.  |                  
                --+ |         +-+     | +--                
-------- 8 7 6 5  | |---------+4+---> | |  3 2 1 --------->
                --+ |    +-+  +-+     | +--                
-------------- 8  | |----+7+--------> | |  6 5 4 3 2 1 --->
                --+ |    +-+    +-+   | +--                
-------- 8 7 6 5  | |-----------+4+-> | |  3 2 1 --------->
                --+ |  +-+      +-+   | +--                
------ 8 7 6 5 4  | |--+3+----------> | |  2 1 ----------->
                --+ |  +-+            | +--                
                  |  '---------------'  |                  
                  +--.---.---.---.---.--+                  
                     |   |   |   |   |     
</pre>

NumBirch supports three different backends based on the following libraries, as well as custom code:

| Backend | Device | Memory            | Linear algebra   | Special math functions | Primitives |
| ------- | ------ | ----------------- | ---------------- | ---------------------- | ---------- |
| Eigen   | CPU    | host, jemalloc    | Eigen            | Eigen                  |            |
| CUDA    | GPU    | unified, jemalloc | cuBLAS, cuSOLVER | Eigen                  | CUB        |
| oneAPI  | GPU    | unified, jemalloc | MKL              | Eigen                  | DPL        |

## License

NumBirch is open source software. It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

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

To use the CUDA backend you will need an Nvidia GPU. You are unlikely to see speedups over the Eigen backend unless your profile is dominated by large matrix operations (of at least hundreds of rows and columns).

In addition to the requirements for the default backend, the CUDA backend requires [CUDA](https://developer.nvidia.com/cuda-downloads) and a custom build of [jemalloc](http://jemalloc.net/). It may also require a more recent version of Eigen compatible with your version of CUDA.

#### 1. Build jemalloc

Run, from within the `jemalloc` directory (or the source directory of a separate jemalloc source package):

```
./configure \
    --enable-static \
    --disable-shared \
    --disable-documentation \
    --disable-initial-exec-tls \
    --with-jemalloc-prefix=numbirch_ \
    --with-install-suffix=_numbirch
make
make install
```

Replace `./configure` with `./autogen.sh` if the former does not exist. If installing system wide, the last command may require `sudo`.

The options `--with-jemalloc-prefix=numbirch_` and `--with-install-suffix=_numbirch` are critical and no other values should be used; NumBirch and some dependencies are hard-coded to work with these values.

#### 2. Install the `nvcc_wrapper` script

The `nvcc_wrapper` script is a thin wrapper around `nvcc` that resolves some incompatibilites between it and the GNU Autotools, specifically around command-line options that lead to incorrect dependency detection and shared library behavior. It must be installed somewhere on your `$PATH` but *outside* the NumBirch source directory to work correctly. It is, however, only required for building NumBirch; developers who expect to rebuild NumBirch frequently may like to install it to a permanent location, users who expect to rebuild infrequently may like to install to a temporary location instead:

    mkdir $HOME/tmpbin
    export OLDPATH=$PATH
    export PATH=$HOME/tmpbin:$OLDPATH
    cp nvcc_wrapper $HOME/tmpbin/.

#### 3. Install NumBirch

Finally, in the root directory of the NumBirch sources, build NumBirch with the CUDA backend:

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

The oneAPI requires the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.g8tsv3). Once installed, run

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

To run using the oneAPI backend you will need an Intel GPU, such as the integrated graphics on an Intel CPU.

## Getting started

Include the header file:

```
#include <numbirch.hpp>
```

and add (at least) `-std=c++14` to the compile line and `-lnumbirch` to the link line.

According to your installation, various libraries may be available that correspond to the different backends, e.g. `libnumbirch-cuda.so` for the CUDA backend. Any of these may be linked instead, and usage does not change.

A simple "Hello world" program is as follows. In `hello.cpp`:

```
#include <numbirch.hpp>
#include <iostream>

int main() {
  Matrix<double> A = {{1.0, 0.5}, {0.0, 1.0}};
  Vector<double> x = {10.0, 2.0};
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

While the interface of NumBirch may look a little complicated with its template code, it works with few surprises. You can declare matrices and vectors:

```
Matrix<double> A;
Vector<double> x;
```

and e.g. multiply them together:

```
auto y = A*x;
```

to get, in this case, `y` of type `Vector<double>`. There are numerous *transformations* for standard math functions that you can batch across arrays, e.g.:

```
auto z = sin(y);
```

to get, again, `z` of type `Vector<double>`. When scalars are required you can use primitive scalar types:

```
auto a = 2.0;
z = 2.0*z;
```

In fact, `Matrix<T>` is just an alias for `Array<T,2>`, `Vector<T>` for `Array<T,1>`, and `Scalar<T>` for `Array<T,0>`. The latter can work like a primitive scalar:

```
Scalar<double> b = 2.0;
z = b*z;
```

but has a role in asynchronous computing, described later.

NumBirch supports four different types for all operations: `bool`, `int`, `float`, `double`. Any function or operator may be called with arguments of these types, or arguments that are arrays of these types. While the NumBirch interface is made up of template functions, these templates are explicitly instantiated for all valid types in the NumBirch library linked when building. This design allows the use of a regular C++ toolchain for compiling client projects, rather than the more complex toolchains required for some of the backends. Much simpler.

### Understanding type traits

Most functions are defined with generic parameter types. The C++ idiom of *SFINAE* ("Substitution Failure Is Not An Error") is used to restrict the accepted types according to *type traits*. Consider, for example, the addition operator. Its full signature is:

```
template<class R, class T, class U, class = std::enable_if_t<
    is_arithmetic_v<R> && is_numeric_v<T> && is_numeric_v<U> &&
    is_compatible_v<T,U>,int>>
convert_t<R,T,U> operator+(const T& x, const U& y);
```

The parameter types are generic: `T` and `U`, but according to the type traits in the template parameter list, the return type `R` must be *arithmetic*, and the argument types `T` and `U` must be *numeric* and mutually *compatible*. The type traits that you may encounter are:

| Name             | Trait                 | Description                                                  |
| ---------------- | :-------------------- | :----------------------------------------------------------- |
| *integral*       | `is_integral_v`       | `bool` or `int`, c.f. [`std::is_integral`](https://en.cppreference.com/w/cpp/types/is_integral) |
| *floating point* | `is_floating_point_v` | `float` or `double`, c.f. [`std::is_floating_point`](https://en.cppreference.com/w/cpp/types/is_integral) |
| *arithmetic*     | `is_arithmetic_v`     | `bool`, `int`, `float`, or `double`, c.f. [`std::is_arithmetic`](https://en.cppreference.com/w/cpp/types/is_integral) |
| *array*          | `is_array_v`          | `Array<T,D>` where `T` is any *arithmetic* type              |
| *numeric*        | `is_numeric_v`        | Any *arithmetic* or *array* type                             |
| *compatible*     | `is_compatible_v`     | The types must have the same number of dimensions. Zero-dimensional arrays (e.g. `Array<double,0>` a.k.a. `Scalar<double>`) are also compatible with scalars (e.g. `double`). |

### Implicit conversion

Arithmetic types promote in the order `bool` to `int` to `float` to `double`.
The return type of an operation between two or more different types will be
the highest type in this promotion order, unless explicitly specified
otherwise; e.g. an operation between a `bool` and `int` yields an `int`,
between a `float` and `double` yields a `double`, between an `int` and `float`
yields a `float`.

The same extends to arrays; e.g. an operation between an `Array<int,D>` and `Array<float,D>` yields an `Array<float,D>`.

@note These numerical promotion rules differ from the broader rules of C++. Specifically, under C++ rules, an operation between an `int` and a `float` promotes to a `double`, not to a `float` as here. The choice to deviate from the C++ rules reflects that working in single precision has a significant performance advantage on much modern hardware, such as GPUs, where the FLOPS ratio between single- and double-precision can be as high as 32:1. Under the C++ rules, it can be a little too easy to promote from single- to double-precision operations unintentionally. The NumBirch rules eliminate some common situations where this occurs.

### Explicit conversion

All functions have two overloads, one that determines the return type implicitly according to the rules described above, and one that requires it explicitly. Take, for example, the function lfact(), which computes the logarithm of the factorial of its argument. The first overload determines the return type implicitly, as being the same as the argument type:

```
template<class T>
T lfact(const T& x);
```

This may be called as, for example, `lfact(10.0)`, and as the argument type is `double`, the return type is also `double`. The second overload requires that the return type is specified explicitly:

```
template<class R, class T>
R lfact(const T& x);
```

This would be called as e.g. `lfact<double>(10)` or `lfact<float>(10)` according to the desired return type. The usual use case for this is precisely for functions like lfact(), which sensibly takes an integral argument, but ought to return a floating point result. The approach can be used to specify arbitrary conversions, however, such as `lfact<float>(10.0)`, where the argument has type `double` but the result type `float`.

@note In these cases, the computation is performed in the return type, i.e. the argument cast to the return type and the computation performed, rather than the computation performed and the result cast to the return type.

Some functions, notably linear algebra functions, accept only matrix and vector arguments of the same floating point type (i.e. all `float` or all `double`). This restriction is maintained by the NumBirch interface to maximize backend compatibility, as not all support mixed types.

### Default conversion

Finally, we may call `lfact(10)`; lacking an explicit return type, NumBirch reverts to its default. The default is `double` unless configured otherwise.

More precisely, the default is `real`, a type defined by NumBirch to be `double`. You can instead define it as `float` by defining the macro `NUMBIRCH_REAL` to be `float`. You can do this either immediately before including the `numbirch.hpp` header file:

```
#define NUMBIRCH_REAL float
#include <numbirch.hpp>
```

Or, to ensure consistency through your sources, define it in your build system, using `-DNUMBIRCH_REAL=float` on the compile line.

The type `real` has other uses. Promotion to `double` via literals is another easy mistake to make. Consider the following code:

```
Array<float,1> x;
auto y = 2.0*x;
```

Here, `y` has type `Array<double,1>`, as the literal `2.0` is of type `double`, causing the promotion. One could instead use a single precision literal `2.0f`, which is of type `float`. Another approach, if one has no intention of using mixed precision arithmetic, is to use `real` throughout, always casting literals to ensure that they are of type `real`:

```
Array<real,1> x;
auto y = real(2.0)*x;
```

Now, you can compile your whole program with `NUMBIRCH_REAL` set to either `float` or `double` to compile single- and double-precision versions, and be confident that operations will not promote to double-precision in the single-precision version.

## Asynchronous computation

NumBirch uses an asynchronous computing model. That asynchronicity is between the *host*, on which your code is running, and the *device* on which NumBirch kernels are launched. The host is the CPU. If using the default backend, the device is also the CPU; in this case computation is synchronous, so that a call to a NumBirch function will not return until its computation is complete. If using e.g. the CUDA backend, the device is a GPU; in this case computation is asynchronous, so that a call to a NumBirch function may return before its computation is complete. In this latter case understanding the asynchronous computing model may be necessary to obtain optimal performance.

The general principal is that *arrays are element-wise futures*. An array returned from one function can be passed as an argument to another function with no synchronization required---the contents of the array may not have been computed yet, but you don't need it yet either, and the backend will stream these function calls one after another when the device is ready. Arrays can even go out of scope and be destroyed without synchronization---memory allocation and deallocation is streamed asynchronously by the backend too, so a deallocation of an array will not occur until all computation involving it has completed.

Synchronization is only required when accessing an array *element-wise*. This is handled internally by the numbirch::Array class, so there is nothing extra for you to do in your own code except be aware that this has performance implications: your code will block until all computation on the device for the current host thread is complete. Element-wise access includes accessing individual elements (called *dicing*, as opposed to *slicing* in the Array idioms), or obtaining an iterator over elements. This idiom avoids synchronization until you are ready to access the result. See numbirch::Array for further details.

In this way, arrays act as *futures* (c.f. [`std::future`](https://en.cppreference.com/w/cpp/thread/future)), providing an easy mechanism to synchronize automatically when necessary. Some functions even return an `Array<T,0>` a.k.a `Scalar<T>` or `Future<T>`. Such a zero-dimensional array has just one element, but works like any other array with respect to being an element-wise future on that element. Typical functions that work this way are those that perform reductions, e.g. sum(), trace(), ldet().

### Thread safety

* Copy on write is thread safe.
* Multiple threads may read and write the same array simultaneously. As usual, there is no guarantee as to the order of operations without synchronization between threads.
* This extends to array destruction. When destroying an array used by multiple threads, there is no guarantee that the destruction occurs after all operations have completed without synchronization between threads. Because memory pools are used, this can compound to another thread recycling the memory for a new array prematurely, before other threads have finished using it for the old array.
* Because of asynchronous computation, host threads may need to synchronize with their respective device on occasion. If using numbirch::Array only, this is handled for you. If using the memory allocation functions directly, you will need to handle this yourself---see documentation for functions such as malloc(), realloc(), and free().

When synchronizing host threads externally, keep in mind that this will not synchronize their respective devices. To do so, each host thread should call wait() to synchronize with its device before synchronizing with other host threads.

If using OpenMP, considering adding wait() at the end of parallel regions for a global barrier across host threads and devices.

## Asynchronous memory management

### Copy on write

All arrays are copy-on-write. This allows multiple arrays to share the same underlying memory for as long as they are only read, with a new buffer allocated and the contents copied over when a write is attempted. There is a small overhead in the use of atomic spin locks when this occurs, but for many use cases, a significant saving in the elimination of unnecessary copies.

### Memory pooling

NumBirch uses [jemalloc](http://jemalloc.net/) with custom extent hooks to allocate unified memory, i.e. memory that can be accessed on both host and device. Separate arenas are used for memory that is used only on the device (e.g. for temporaries within a single numeric function) versus that which may be used on both (e.g. for the buffers used by numbirch::Array). In all cases, however, unified memory is used, as jemalloc itself may need to access the memory on host. Each arena has its own thread-local memory pool. Memory allocations (e.g. via `cudaMallocManaged()`) occur either to extend the size of the extents used by pools, or for very large allocations. These tend to occur early on in program execution and their frequency diminish in time, although this will depend on the memory profile of the particular program.

# Q&A

### Does NumBirch support mixed-precision floating point operations?

No.

Firstly, backends often do not support mixed precision for certain operations (consider e.g. BLAS and LAPACK interfaces). Secondly, the need to support forward (evaluation) and backward (gradient) computations for reverse-mode automatic differentiation means that floating point promotion occurs in both directions, causing incompatibilities with those backends one way or the other. The choice was made to avoid this complication, at least for now.
