lib_LTLIBRARIES =
if DEBUG
lib_LTLIBRARIES += libnumbirch-debug.la
if SINGLE
lib_LTLIBRARIES += libnumbirch-debug-single.la
endif
endif
if TEST
lib_LTLIBRARIES += libnumbirch-test.la
if SINGLE
lib_LTLIBRARIES += libnumbirch-test-single.la
endif
endif
if RELEASE
lib_LTLIBRARIES += libnumbirch.la
if SINGLE
lib_LTLIBRARIES += libnumbirch-single.la
endif
endif

AM_CPPFLAGS = 
AM_CXXFLAGS = 
AM_LDFLAGS = 

SOURCES =  \
    numbirch/common/random.cpp \
    numbirch/instantiate/array/diagonal.cpp \
    numbirch/instantiate/array/element_matrix.cpp \
    numbirch/instantiate/array/element_vector.cpp \
    numbirch/instantiate/array/mat.cpp \
    numbirch/instantiate/array/single_matrix.cpp \
    numbirch/instantiate/array/single_vector.cpp \
    numbirch/instantiate/array/vec.cpp \
    numbirch/instantiate/memory/memcpy.cpp \
    numbirch/instantiate/memory/memset.cpp \
    numbirch/instantiate/numeric/binary_matrix.cpp \
    numbirch/instantiate/numeric/dot.cpp \
    numbirch/instantiate/numeric/frobenius.cpp \
    numbirch/instantiate/numeric/inner.cpp \
    numbirch/instantiate/numeric/matrix.cpp \
    numbirch/instantiate/numeric/outer.cpp \
    numbirch/instantiate/numeric/reduce_matrix.cpp \
    numbirch/instantiate/numeric/triouter.cpp \
    numbirch/instantiate/random/binary.cpp \
    numbirch/instantiate/random/standard_gaussian.cpp \
    numbirch/instantiate/random/standard_wishart.cpp \
    numbirch/instantiate/random/unary.cpp \
    numbirch/instantiate/reduce/count.cpp \
    numbirch/instantiate/reduce/count_grad.cpp \
    numbirch/instantiate/reduce/sum.cpp \
    numbirch/instantiate/reduce/sum_grad.cpp \
    numbirch/instantiate/transform/binary.cpp \
    numbirch/instantiate/transform/binary_grad.cpp \
    numbirch/instantiate/transform/binary_operator.cpp \
    numbirch/instantiate/transform/cast.cpp \
    numbirch/instantiate/transform/ternary.cpp \
    numbirch/instantiate/transform/unary.cpp \
    numbirch/instantiate/transform/unary_grad.cpp \
    numbirch/instantiate/transform/unary_operator.cpp

if ONEAPI
  AM_CPPFLAGS += -DBACKEND_ONEAPI \
      -Wno-deprecated-declarations \
      -Wno-unused-command-line-argument
  AM_CXXFLAGS += -fsycl -qmkl
  AM_LDFLAGS += $(AM_CXXFLAGS)
  SOURCES += \
      numbirch/jemalloc/memory.cpp \
      numbirch/oneapi/memory.cpp \
      numbirch/oneapi/sycl.cpp
else
if CUDA
  AM_CPPFLAGS += -DBACKEND_CUDA
  AM_CXXFLAGS += -x cu -G \
      -default-stream=per-thread \
      -expt-relaxed-constexpr \
      -arch=sm_70 \
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
      numbirch/jemalloc/memory.cpp
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
  numbirch/array/Future.hpp \
  numbirch/array/Lock.hpp \
  numbirch/array/Matrix.hpp \
  numbirch/array/Scalar.hpp \
  numbirch/array/Vector.hpp \
  numbirch/array.hpp \
  numbirch/memory.hpp \
  numbirch/numbirch.hpp \
  numbirch/numeric.hpp \
  numbirch/random.hpp \
  numbirch/reduce.hpp \
  numbirch/transform.hpp \
  numbirch/utility.hpp

noinst_HEADERS = \
  numbirch/common/array.inl \
  numbirch/common/random.hpp \
  numbirch/common/random.inl \
  numbirch/common/reduce.inl \
  numbirch/common/transform.inl \
  numbirch/cuda/cub.hpp \
  numbirch/cuda/cublas.hpp \
  numbirch/cuda/cuda.hpp \
  numbirch/cuda/curand.hpp \
  numbirch/cuda/cusolver.hpp \
  numbirch/cuda/memory.inl \
  numbirch/cuda/numeric.inl \
  numbirch/cuda/random.inl \
  numbirch/cuda/reduce.inl \
  numbirch/cuda/transform.inl \
  numbirch/eigen/eigen.hpp \
  numbirch/eigen/memory.inl \
  numbirch/eigen/numeric.inl \
  numbirch/eigen/random.inl \
  numbirch/eigen/reduce.inl \
  numbirch/eigen/transform.inl \
  numbirch/jemalloc/jemalloc.hpp \
  numbirch/oneapi/dpl.hpp \
  numbirch/oneapi/numeric.hpp \
  numbirch/oneapi/sycl.hpp

dist_noinst_DATA =  \
  Doxyfile \
  dpcpp_wrapper \
  LICENSE
