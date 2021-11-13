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
    numbirch/numeric/binary.cpp \
    numbirch/numeric/other.cpp \
    numbirch/numeric/ternary.cpp \
    numbirch/numeric/unary.cpp \
    numbirch/random.cpp

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
  AM_CXXFLAGS += -x cu -G -default-stream=per-thread -expt-relaxed-constexpr -Xcompiler="-Wall" -Xcompiler="-Wno-unused-result" -Xcompiler="-Wno-unknown-pragmas" -Xcompiler="-Wno-deprecated-declarations"
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
  numbirch/memory.hpp \
  numbirch/numbirch.hpp \
  numbirch/numeric.hpp \
  numbirch/random.hpp \
  numbirch/type.hpp

noinst_HEADERS = \
  numbirch/cuda/cub.hpp \
  numbirch/cuda/cublas.hpp \
  numbirch/cuda/cuda.hpp \
  numbirch/cuda/cusolver.hpp \
  numbirch/cuda/numeric.hpp \
  numbirch/eigen/eigen.hpp \
  numbirch/eigen/numeric.hpp \
  numbirch/jemalloc/jemalloc.hpp \
  numbirch/oneapi/dpl.hpp \
  numbirch/oneapi/numeric.hpp \
  numbirch/oneapi/sycl.hpp \
  numbirch/functor.hpp \
  numbirch/macro.hpp

dist_noinst_DATA =  \
  Doxyfile \
  dpcpp_wrapper \
  LICENSE
