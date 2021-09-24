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

AM_CPPFLAGS = -Wall

libbirch_debug_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O -g -flto
libbirch_debug_la_LIBADD = $(DEBUG_LIBS)
libbirch_debug_la_SOURCES = $(COMMON_SOURCES)

libbirch_test_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O -g -flto --coverage
libbirch_test_la_LIBADD = $(TEST_LIBS)
libbirch_test_la_SOURCES = $(COMMON_SOURCES)

libbirch_la_CPPFLAGS = $(AM_CPPFLAGS) -DNDEBUG
libbirch_la_CXXFLAGS = $(AM_CXXFLAGS) $(OPENMP_CXXFLAGS) -O3 -flto
libbirch_la_LIBADD = $(RELEASE_LIBS)
libbirch_la_SOURCES = $(COMMON_SOURCES)

include_HEADERS = \
  libbirch/libbirch.hpp

nobase_include_HEADERS = \
  libbirch/Any.hpp \
  libbirch/Atomic.hpp \
  libbirch/BiconnectedCollector.hpp \
  libbirch/BiconnectedCopier.hpp \
  libbirch/BiconnectedMemo.hpp \
  libbirch/Bridger.hpp \
  libbirch/Collector.hpp \
  libbirch/Copier.hpp \
  libbirch/Destroyer.hpp \
  libbirch/docs.hpp \
  libbirch/external.hpp \
  libbirch/internal.hpp \
  libbirch/Marker.hpp \
  libbirch/macro.hpp \
  libbirch/Memo.hpp \
  libbirch/memory.hpp \
  libbirch/mutable.hpp \
  libbirch/Reacher.hpp \
  libbirch/Scanner.hpp \
  libbirch/Shared.hpp \
  libbirch/Spanner.hpp \
  libbirch/thread.hpp \
  libbirch/type.hpp

COMMON_SOURCES =  \
  libbirch/Any.cpp \
  libbirch/BiconnectedMemo.cpp \
  libbirch/Memo.cpp \
  libbirch/memory.cpp

dist_noinst_DATA =  \
  Doxyfile \
  LICENSE
