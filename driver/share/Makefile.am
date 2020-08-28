# compile flags
AM_CPPFLAGS = -Wall -DEIGEN_NO_STATIC_ASSERT -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1
COMMON_CXXFLAGS = -include birch_PACKAGE_TARNAME.hpp $(OPENMP_CXXFLAGS)
if DEBUG
AM_CXXFLAGS = $(COMMON_CXXFLAGS) -O0 -g -fno-inline
else
if TEST
AM_CXXFLAGS = $(COMMON_CXXFLAGS) -O0 -g -fno-inline -fprofile-abs-path --coverage
else
if RELEASE
AM_CXXFLAGS = $(COMMON_CXXFLAGS) -O3 -g
else
AM_CXXFLAGS = $(COMMON_CXXFLAGS)
endif
endif
endif

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
