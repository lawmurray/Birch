lib_LTLIBRARIES =
if DEBUG
lib_LTLIBRARIES += libbirch-debug.la
endif
if TEST
lib_LTLIBRARIES += libbirch-test.la
endif
if RELEASE
lib_LTLIBRARIES += libbirch.la
endif

AM_CPPFLAGS = -Wall -DEIGEN_NO_STATIC_ASSERT -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1
if MEMORY_POOL
AM_CPPFLAGS += -DENABLE_MEMORY_POOL=1
endif

libbirch_debug_la_CXXFLAGS = $(OPENMP_CXXFLAGS) -O0 -g -fno-inline
libbirch_debug_la_SOURCES = $(COMMON_SOURCES)

libbirch_test_la_CXXFLAGS = $(OPENMP_CXXFLAGS) -O0 -g -fno-inline --coverage
libbirch_test_la_SOURCES = $(COMMON_SOURCES)

libbirch_la_CPPFLAGS = -DNDEBUG
libbirch_la_CXXFLAGS = $(OPENMP_CXXFLAGS) -O3
libbirch_la_SOURCES = $(COMMON_SOURCES)

include_HEADERS = \
  libbirch/libbirch.hpp

nobase_include_HEADERS = \
  libbirch/Allocator.hpp \
  libbirch/Any.hpp \
  libbirch/Array.hpp \
  libbirch/Atomic.hpp \
  libbirch/assert.hpp \
  libbirch/Bridger.hpp \
  libbirch/Buffer.hpp \
  libbirch/Collector.hpp \
  libbirch/Copier.hpp \
  libbirch/docs.hpp \
  libbirch/Dimension.hpp \
  libbirch/Eigen.hpp \
  libbirch/external.hpp \
  libbirch/Index.hpp \
  libbirch/internal.hpp \
  libbirch/Iterator.hpp \
  libbirch/Length.hpp \
  libbirch/Lock.hpp \
  libbirch/Marker.hpp \
  libbirch/macro.hpp \
  libbirch/memory.hpp \
  libbirch/mutable.hpp \
  libbirch/Offset.hpp \
  libbirch/Pool.hpp \
  libbirch/Range.hpp \
  libbirch/Reacher.hpp \
  libbirch/ReadersWriterLock.hpp \
  libbirch/Scanner.hpp \
  libbirch/Shape.hpp \
  libbirch/Shared.hpp \
  libbirch/Slice.hpp \
  libbirch/Spanner.hpp \
  libbirch/stacktrace.hpp \
  libbirch/Stride.hpp \
  libbirch/thread.hpp \
  libbirch/type.hpp

COMMON_SOURCES =  \
  libbirch/Bridger.cpp \
  libbirch/Copier.cpp \
  libbirch/Collector.cpp \
  libbirch/Marker.cpp \
  libbirch/Reacher.cpp \
  libbirch/Scanner.cpp \
  libbirch/Spanner.cpp \
  libbirch/memory.cpp \
  libbirch/stacktrace.cpp

dist_noinst_DATA =  \
  Doxyfile \
  LICENSE
