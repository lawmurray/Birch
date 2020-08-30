lib_LTLIBRARIES =
if DEBUG
lib_LTLIBRARIES += libPACKAGE_TARNAME-debug.la
endif
if TEST
lib_LTLIBRARIES += libPACKAGE_TARNAME-test.la
endif
if RELEASE
lib_LTLIBRARIES += libPACKAGE_TARNAME.la
endif

AM_CPPFLAGS = -Wall -DEIGEN_NO_STATIC_ASSERT -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1
COMMON_CXXFLAGS = -include PACKAGE_TARNAME.hpp $(OPENMP_CXXFLAGS)

libPACKAGE_TARNAME_debug_la_CXXFLAGS = $(COMMON_CXXFLAGS) -O0 -g -fno-inline
libPACKAGE_TARNAME_debug_la_LIBADD = $(DEBUG_LIBS)
libPACKAGE_TARNAME_debug_la_SOURCES = $(COMMON_SOURCES)

libPACKAGE_TARNAME_test_la_CXXFLAGS = $(COMMON_CXXFLAGS) -O0 -g -fno-inline -fprofile-abs-path --coverage
libPACKAGE_TARNAME_test_la_LIBADD = $(TEST_LIBS)
libPACKAGE_TARNAME_test_la_SOURCES = $(COMMON_SOURCES)

libPACKAGE_TARNAME_la_CPPFLAGS = -DNDEBUG
libPACKAGE_TARNAME_la_CXXFLAGS = $(COMMON_CXXFLAGS) -O3
libPACKAGE_TARNAME_la_LIBADD = $(RELEASE_LIBS)
libPACKAGE_TARNAME_la_SOURCES = $(COMMON_SOURCES)

# placeholder source file for Emscripten program wrapper
main.cpp:
	touch main.cpp

# Emscripten program wrapper
EXTRA_PROGRAMS = birch.js birch.html

birch_js_LDADD = libbirch_PACKAGE_TARNAME.la
birch_js_SOURCES = main.cpp
birch_js_LDFLAGS = -s MAIN_MODULE=1 --emrun

birch_html_LDADD = libbirch_PACKAGE_TARNAME.la
birch_html_SOURCES = main.cpp
birch_html_LDFLAGS = -s MAIN_MODULE=1 --emrun

BUILT_SOURCES = main.cpp
CLEANFILES = $(BUILT_SOURCES)
