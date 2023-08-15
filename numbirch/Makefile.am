AM_LDFLAGS = -release $(PACKAGE_VERSION)
AM_CXXFLAGS = -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS=1
# ^ BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS set to avoid valgrind issues

SOURCES =  \
    numbirch/common/random.cpp \
    numbirch/instantiate/array.cpp \
    numbirch/instantiate/memory.cpp \
    numbirch/instantiate/numeric.cpp \
    numbirch/instantiate/random.cpp \
    numbirch/instantiate/reduce.cpp \
    numbirch/instantiate/transform.cpp

ONEAPI_CPPFLAGS = -DBACKEND_ONEAPI \
    -Wno-deprecated-declarations \
    -Wno-unused-command-line-argument
ONEAPI_CXXFLAGS = -fsycl -qmkl
ONEAPI_LDFLAGS = $(AM_CXXFLAGS)
ONEAPI_SOURCES = \
    numbirch/jemalloc/memory.cpp \
    numbirch/oneapi/memory.cpp \
    numbirch/oneapi/sycl.cpp

CUDA_CPPFLAGS = -DBACKEND_CUDA
CUDA_CXXFLAGS = -x cu \
    -expt-relaxed-constexpr \
    -arch=sm_70 \
    -Xcompiler="-Wno-unused-result" \
    -Xcompiler="-Wno-unknown-pragmas" \
    -Xcompiler="-Wno-cpp" \
    -Xcompiler="-Wno-deprecated-declarations"
CUDA_SOURCES = \
    numbirch/cuda/cublas.cpp \
    numbirch/cuda/cuda.cpp \
    numbirch/cuda/curand.cpp \
    numbirch/cuda/cusolver.cpp \
    numbirch/cuda/memory.cpp \
    numbirch/cuda/random.cpp \
    numbirch/jemalloc/memory.cpp

EIGEN_CPPFLAGS = -DBACKEND_EIGEN \
    -DEIGEN_NO_AUTOMATIC_RESIZING=1 \
    -DEIGEN_DONT_PARALLELIZE=1
EIGEN_SOURCES = \
    numbirch/eigen/memory.cpp \
    numbirch/eigen/random.cpp

lib_LTLIBRARIES =

if ONEAPI
if SINGLE
lib_LTLIBRARIES += libnumbirch-oneapi-single.la
endif
if DOUBLE
lib_LTLIBRARIES += libnumbirch-oneapi.la
endif
endif

if CUDA
if SINGLE
lib_LTLIBRARIES += libnumbirch-cuda-single.la
endif
if DOUBLE
lib_LTLIBRARIES += libnumbirch-cuda.la
endif
endif

if EIGEN
if SINGLE
lib_LTLIBRARIES += libnumbirch-single.la
endif
if DOUBLE
lib_LTLIBRARIES += libnumbirch.la
endif
endif

libnumbirch_oneapi_la_CPPFLAGS = $(AM_CPPFLAGS) $(ONEAPI_CPPFLAGS) -DNUMBIRCH_REAL=double
libnumbirch_oneapi_la_CXXFLAGS = $(AM_CXXFLAGS) $(ONEAPI_CXXFLAGS)
libnumbirch_oneapi_la_LDFLAGS = $(AM_LDFLAGS) $(ONEAPI_LDFLAGS)
libnumbirch_oneapi_la_SOURCES = $(SOURCES) $(ONEAPI_SOURCES)

libnumbirch_oneapi_single_la_CPPFLAGS = $(AM_CPPFLAGS) $(ONEAPI_CPPFLAGS) -DNUMBIRCH_REAL=float
libnumbirch_oneapi_single_la_CXXFLAGS = $(AM_CXXFLAGS) $(ONEAPI_CXXFLAGS)
libnumbirch_oneapi_single_la_LDFLAGS = $(AM_LDFLAGS) $(ONEAPI_LDFLAGS)
libnumbirch_oneapi_single_la_SOURCES = $(SOURCES) $(ONEAPI_SOURCES)

libnumbirch_cuda_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CPPFLAGS) -DNUMBIRCH_REAL=double
libnumbirch_cuda_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CXXFLAGS)
libnumbirch_cuda_la_LDFLAGS = $(AM_LDFLAGS) $(CUDA_LDFLAGS)
libnumbirch_cuda_la_SOURCES = $(SOURCES) $(CUDA_SOURCES)

libnumbirch_cuda_single_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CPPFLAGS) -DNUMBIRCH_REAL=float
libnumbirch_cuda_single_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CXXFLAGS)
libnumbirch_cuda_single_la_LDFLAGS = $(AM_LDFLAGS) $(CUDA_LDFLAGS)
libnumbirch_cuda_single_la_SOURCES = $(SOURCES) $(CUDA_SOURCES)

libnumbirch_la_CPPFLAGS = $(AM_CPPFLAGS) $(EIGEN_CPPFLAGS) -DNUMBIRCH_REAL=double
libnumbirch_la_CXXFLAGS = $(AM_CXXFLAGS) $(EIGEN_CXXFLAGS)
libnumbirch_la_LDFLAGS = $(AM_LDFLAGS) $(EIGEN_LDFLAGS)
libnumbirch_la_SOURCES = $(SOURCES) $(EIGEN_SOURCES)

libnumbirch_single_la_CPPFLAGS = $(AM_CPPFLAGS) $(EIGEN_CPPFLAGS) -DNUMBIRCH_REAL=float
libnumbirch_single_la_CXXFLAGS = $(AM_CXXFLAGS) $(EIGEN_CXXFLAGS)
libnumbirch_single_la_LDFLAGS = $(AM_LDFLAGS) $(EIGEN_LDFLAGS)
libnumbirch_single_la_SOURCES = $(SOURCES) $(EIGEN_SOURCES)

include_HEADERS = numbirch/numbirch.hpp

nobase_include_HEADERS = \
  numbirch/array/Array.hpp \
  numbirch/array/Future.hpp \
  numbirch/array/Iterator.hpp \
  numbirch/array/Lock.hpp \
  numbirch/array/Matrix.hpp \
  numbirch/array/Scalar.hpp \
  numbirch/array/Shape.hpp \
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
  numbirch/instantiate/instantiate.hpp \
  numbirch/jemalloc/jemalloc.hpp \
  numbirch/oneapi/dpl.hpp \
  numbirch/oneapi/numeric.hpp \
  numbirch/oneapi/sycl.hpp

dist_noinst_DATA =  \
  Doxyfile \
  LICENSE \
  CONTRIBUTING.md \
  README.md
