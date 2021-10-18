lib_LTLIBRARIES =
if DEBUG
lib_LTLIBRARIES += libnumbirch-debug.la
endif
if TEST
lib_LTLIBRARIES += libnumbirch-test.la
endif
if RELEASE
lib_LTLIBRARIES += libnumbirch.la
endif

AM_CPPFLAGS = 
AM_CXXFLAGS = 
AM_LDFLAGS = 

COMMON_SOURCES = \
    numbirch/common/numeric/binary_function.cpp \
    numbirch/common/numeric/binary_operator.cpp \
    numbirch/common/numeric/other_function.cpp \
    numbirch/common/numeric/ternary_function.cpp \
    numbirch/common/numeric/ternary_operator.cpp \
    numbirch/common/numeric/unary_function.cpp \
    numbirch/common/numeric/unary_operator.cpp \
    numbirch/common/random.cpp

if ONEAPI
  AM_CPPFLAGS += -DBACKEND_ONEAPI -Wno-deprecated-declarations -Wno-unused-command-line-argument
  AM_CXXFLAGS += -fsycl -qmkl
  AM_LDFLAGS += $(AM_CXXFLAGS)
  COMMON_SOURCES += \
      numbirch/jemalloc/jemalloc.cpp \
      numbirch/oneapi/memory.cpp \
      numbirch/oneapi/sycl.cpp
else
if CUDA
  AM_CPPFLAGS += -DBACKEND_CUDA 
  AM_CXXFLAGS += -x cu -G -arch=sm_75 -expt-relaxed-constexpr -default-stream=per-thread -Xcompiler="-Wall" -Xcompiler="-Wno-unused-result"
  COMMON_SOURCES += \
      numbirch/cuda/cublas.cpp \
      numbirch/cuda/cuda.cpp \
      numbirch/cuda/cusolver.cpp \
      numbirch/cuda/memory.cpp \
      numbirch/jemalloc/jemalloc.cpp
else
  AM_CPPFLAGS += -DBACKEND_EIGEN -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1
  COMMON_SOURCES += \
      numbirch/eigen/memory.cpp
endif
endif

#
# Notes on compiler differences:
#   - nvcc does not support -O without a number, must be e.g. -O1
#

libnumbirch_debug_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O1 -g
libnumbirch_debug_la_SOURCES = $(COMMON_SOURCES)

libnumbirch_test_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O1 -g --coverage
libnumbirch_test_la_SOURCES = $(COMMON_SOURCES)

libnumbirch_la_CPPFLAGS = $(AM_CPPFLAGS) -DNDEBUG
libnumbirch_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O3
libnumbirch_la_SOURCES = $(COMMON_SOURCES)

include_HEADERS = numbirch/numbirch.hpp

nobase_include_HEADERS = \
  numbirch/array/numeric/binary_function.hpp \
  numbirch/array/numeric/binary_operator.hpp \
  numbirch/array/numeric/other_function.hpp \
  numbirch/array/numeric/ternary_function.hpp \
  numbirch/array/numeric/ternary_operator.hpp \
  numbirch/array/numeric/unary_function.hpp \
  numbirch/array/numeric/unary_operator.hpp \
  numbirch/array/Array.hpp \
  numbirch/array/ArrayControl.hpp \
  numbirch/array/ArrayIterator.hpp \
  numbirch/array/ArrayShape.hpp \
  numbirch/array/Atomic.hpp \
  numbirch/array/external.hpp \
  numbirch/array/Future.hpp \
  numbirch/array/Lock.hpp \
  numbirch/array/Matrix.hpp \
  numbirch/array/random.hpp \
  numbirch/array/Scalar.hpp \
  numbirch/array/Vector.hpp \
  numbirch/numeric/binary_function.hpp \
  numbirch/numeric/binary_operator.hpp \
  numbirch/numeric/other_function.hpp \
  numbirch/numeric/ternary_function.hpp \
  numbirch/numeric/ternary_operator.hpp \
  numbirch/numeric/unary_function.hpp \
  numbirch/numeric/unary_operator.hpp \
  numbirch/array.hpp \
  numbirch/function.hpp \
  numbirch/macro.hpp \
  numbirch/memory.hpp \
  numbirch/numbirch.hpp \
  numbirch/numeric.hpp \
  numbirch/random.hpp

noinst_HEADERS = \
  numbirch/cuda/numeric/binary_function.hpp \
  numbirch/cuda/numeric/binary_operator.hpp \
  numbirch/cuda/numeric/other_function.hpp \
  numbirch/cuda/numeric/ternary_function.hpp \
  numbirch/cuda/numeric/ternary_operator.hpp \
  numbirch/cuda/numeric/unary_function.hpp \
  numbirch/cuda/numeric/unary_operator.hpp \
  numbirch/cuda/cub.hpp \
  numbirch/cuda/cublas.hpp \
  numbirch/cuda/cuda.hpp \
  numbirch/cuda/cusolver.hpp \
  numbirch/eigen/numeric/binary_function.hpp \
  numbirch/eigen/numeric/binary_operator.hpp \
  numbirch/eigen/numeric/other_function.hpp \
  numbirch/eigen/numeric/ternary_function.hpp \
  numbirch/eigen/numeric/ternary_operator.hpp \
  numbirch/eigen/numeric/unary_function.hpp \
  numbirch/eigen/numeric/unary_operator.hpp \
  numbirch/eigen/eigen.hpp \
  numbirch/functor/binary_function.hpp \
  numbirch/functor/binary_operator.hpp \
  numbirch/functor/other_function.hpp \
  numbirch/functor/ternary_function.hpp \
  numbirch/functor/ternary_operator.hpp \
  numbirch/functor/unary_function.hpp \
  numbirch/functor/unary_operator.hpp \
  numbirch/jemalloc/jemalloc.hpp \
  numbirch/oneapi/numeric/binary_function.hpp \
  numbirch/oneapi/numeric/binary_operator.hpp \
  numbirch/oneapi/numeric/other_function.hpp \
  numbirch/oneapi/numeric/ternary_function.hpp \
  numbirch/oneapi/numeric/ternary_operator.hpp \
  numbirch/oneapi/numeric/unary_function.hpp \
  numbirch/oneapi/numeric/unary_operator.hpp \
  numbirch/oneapi/dpl.hpp \
  numbirch/oneapi/sycl.hpp \
  numbirch/docs.hpp

dist_noinst_DATA =  \
  Doxyfile \
  dpcpp_wrapper \
  LICENSE \
  nvcc_wrapper
