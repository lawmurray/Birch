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

COMMON_SOURCES =
if ONEAPI
  AM_CPPFLAGS += -Wno-deprecated-declarations -Wno-unused-command-line-argument
  AM_CXXFLAGS += -qmkl -fsycl-unnamed-lambda
  AM_LDFLAGS += $(AM_CXXFLAGS)
  COMMON_SOURCES += numbirch/oneapi/numbirch.cpp
else
if CUDA
  AM_CXXFLAGS += -x cu -arch=sm_75 -G -Xcompiler="-Wall" -Xcompiler="-Wno-unused-result"
  COMMON_SOURCES += numbirch/cuda/numbirch.cpp
else
  AM_CPPFLAGS += -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1
  COMMON_SOURCES += numbirch/default/numbirch.cpp
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

nobase_include_HEADERS =

dist_noinst_DATA =  \
  Doxyfile \
  LICENSE
