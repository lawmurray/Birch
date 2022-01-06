lib_LTLIBRARIES =
if DEBUG
lib_LTLIBRARIES += libnumbirch-debug.la
lib_LTLIBRARIES += libnumbirch-debug-single.la
endif
if TEST
lib_LTLIBRARIES += libnumbirch-test.la
lib_LTLIBRARIES += libnumbirch-test-single.la
endif
if RELEASE
lib_LTLIBRARIES += libnumbirch.la
lib_LTLIBRARIES += libnumbirch-single.la
endif

AM_CPPFLAGS = 
AM_CXXFLAGS = 
AM_LDFLAGS = 

SOURCES =  \
    numbirch/common/stl.cpp \
    numbirch/instantiate/array_diagonal.cpp \
    numbirch/instantiate/array_element_matrix.cpp \
    numbirch/instantiate/array_element_vector.cpp \
    numbirch/instantiate/array_single_matrix.cpp \
    numbirch/instantiate/array_single_vector.cpp \
    numbirch/instantiate/binary_function.cpp \
    numbirch/instantiate/binary_grad.cpp \
    numbirch/instantiate/binary_operator.cpp \
    numbirch/instantiate/binary_scalar_divide.cpp \
    numbirch/instantiate/binary_scalar_multiply.cpp \
    numbirch/instantiate/numeric_dot.cpp \
    numbirch/instantiate/numeric_frobenius.cpp \
    numbirch/instantiate/numeric_matrix.cpp \
    numbirch/instantiate/numeric_matrix_matrix.cpp \
    numbirch/instantiate/numeric_matrix_vector.cpp \
    numbirch/instantiate/numeric_outer.cpp \
    numbirch/instantiate/numeric_reduce_matrix.cpp \
    numbirch/instantiate/random_standard_matrix.cpp \
    numbirch/instantiate/random_standard_vector.cpp \
    numbirch/instantiate/random_standard_wishart.cpp \
    numbirch/instantiate/reduce.cpp \
    numbirch/instantiate/reduce_grad.cpp \
    numbirch/instantiate/ternary_function.cpp \
    numbirch/instantiate/unary_function.cpp \
    numbirch/instantiate/unary_grad.cpp \
    numbirch/instantiate/unary_operator.cpp

if ONEAPI
  AM_CPPFLAGS += -DBACKEND_ONEAPI \
      -Wno-deprecated-declarations \
      -Wno-unused-command-line-argument
  AM_CXXFLAGS += -fsycl -qmkl
  AM_LDFLAGS += $(AM_CXXFLAGS)
  SOURCES += \
      numbirch/jemalloc/jemalloc.cpp \
      numbirch/oneapi/memory.cpp \
      numbirch/oneapi/sycl.cpp
else
if CUDA
  AM_CPPFLAGS += -DBACKEND_CUDA
  AM_CXXFLAGS += -x cu -G \
      -default-stream=per-thread \
      -expt-relaxed-constexpr \
      -arch=sm_70 \
      -Wno-deprecated-gpu-targets \
      -Xcompiler="-Wall" \
      -Xcompiler="-Wno-unused-result" \
      -Xcompiler="-Wno-unknown-pragmas" \
      -Xcompiler="-Wno-deprecated-declarations"
  SOURCES += \
      numbirch/cuda/cublas.cpp \
      numbirch/cuda/cuda.cpp \
      numbirch/cuda/curand.cpp \
      numbirch/cuda/cusolver.cpp \
      numbirch/cuda/memory.cpp \
      numbirch/cuda/random.cpp \
      numbirch/jemalloc/jemalloc.cpp
else
  AM_CPPFLAGS += -DBACKEND_EIGEN \
      -DEIGEN_NO_AUTOMATIC_RESIZING=1 \
      -DEIGEN_DONT_PARALLELIZE=1
  SOURCES += \
      numbirch/eigen/memory.cpp \
      numbirch/eigen/random.cpp
endif
endif

#
# Notes on compiler differences:
#   - nvcc does not support -O without a number, must be e.g. -O1
#

libnumbirch_debug_la_CPPFLAGS = $(AM_CPPFLAGS)
libnumbirch_debug_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O1 -g
libnumbirch_debug_la_SOURCES = $(SOURCES)
libnumbirch_debug_single_la_CPPFLAGS = $(AM_CPPFLAGS) -DNUMBIRCH_REAL=float
libnumbirch_debug_single_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O1 -g
libnumbirch_debug_single_la_SOURCES = $(SOURCES)

libnumbirch_test_la_CPPFLAGS = $(AM_CPPFLAGS)
libnumbirch_test_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O1 -g --coverage
libnumbirch_test_la_SOURCES = $(SOURCES)
libnumbirch_test_single_la_CPPFLAGS = $(AM_CPPFLAGS) -DNUMBIRCH_REAL=float
libnumbirch_test_single_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O1 -g --coverage
libnumbirch_test_single_la_SOURCES = $(SOURCES)

libnumbirch_la_CPPFLAGS = $(AM_CPPFLAGS) -DNDEBUG
libnumbirch_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O3
libnumbirch_la_SOURCES = $(SOURCES)
libnumbirch_single_la_CPPFLAGS = $(AM_CPPFLAGS) -DNDEBUG -DNUMBIRCH_REAL=float
libnumbirch_single_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O3
libnumbirch_single_la_SOURCES = $(SOURCES)

include_HEADERS = numbirch/numbirch.hpp

nobase_include_HEADERS = \
  numbirch/array/Array.hpp \
  numbirch/array/ArrayControl.hpp \
  numbirch/array/ArrayIterator.hpp \
  numbirch/array/ArrayShape.hpp \
  numbirch/array/Atomic.hpp \
  numbirch/array/external.hpp \
  numbirch/array/Future.hpp \
  numbirch/array/Lock.hpp \
  numbirch/array/Matrix.hpp \
  numbirch/array/Scalar.hpp \
  numbirch/array/Vector.hpp \
  numbirch/array.hpp \
  numbirch/binary.hpp \
  numbirch/macro.hpp \
  numbirch/memory.hpp \
  numbirch/numbirch.hpp \
  numbirch/numeric.hpp \
  numbirch/random.hpp \
  numbirch/reduce.hpp \
  numbirch/ternary.hpp \
  numbirch/type.hpp \
  numbirch/unary.hpp

noinst_HEADERS = \
  numbirch/common/array.hpp \
  numbirch/common/binary.hpp \
  numbirch/common/element.hpp \
  numbirch/common/functor.hpp \
  numbirch/common/random.hpp \
  numbirch/common/reduce.hpp \
  numbirch/common/stl.hpp \
  numbirch/common/ternary.hpp \
  numbirch/common/unary.hpp \
  numbirch/cuda/cub.hpp \
  numbirch/cuda/cublas.hpp \
  numbirch/cuda/cuda.hpp \
  numbirch/cuda/curand.hpp \
  numbirch/cuda/cusolver.hpp \
  numbirch/cuda/numeric.hpp \
  numbirch/cuda/random.hpp \
  numbirch/cuda/reduce.hpp \
  numbirch/cuda/transform.hpp \
  numbirch/eigen/eigen.hpp \
  numbirch/eigen/numeric.hpp \
  numbirch/eigen/random.hpp \
  numbirch/eigen/reduce.hpp \
  numbirch/eigen/transform.hpp \
  numbirch/jemalloc/jemalloc.hpp \
  numbirch/oneapi/dpl.hpp \
  numbirch/oneapi/numeric.hpp \
  numbirch/oneapi/sycl.hpp

dist_noinst_DATA =  \
  Doxyfile \
  dpcpp_wrapper \
  LICENSE
