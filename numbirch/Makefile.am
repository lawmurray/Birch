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
  AM_CPPFLAGS +=
  AM_CXXFLAGS += -qmkl=sequential -Wno-unused-command-line-argument
  AM_LDFLAGS += -lmkl_tbb_thread -lmkl_core -qmkl=sequential
  COMMON_SOURCES += numbirch/numbirch_oneapi.cpp
else
if CUDA
  COMMON_SOURCES += numbirch/numbirch_cuda.cu
else
  AM_CPPFLAGS += -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1
  COMMON_SOURCES += numbirch/numbirch_default.cpp
endif
endif

libnumbirch_debug_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O -g
libnumbirch_debug_la_SOURCES = $(COMMON_SOURCES)

libnumbirch_test_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O -g --coverage
libnumbirch_test_la_SOURCES = $(COMMON_SOURCES)

libnumbirch_la_CPPFLAGS = $(AM_CPPFLAGS) -DNDEBUG
libnumbirch_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O3
libnumbirch_la_SOURCES = $(COMMON_SOURCES)

include_HEADERS = \
  numbirch/numbirch.hpp

nobase_include_HEADERS =

dist_noinst_DATA =  \
  Doxyfile \
  LICENSE
