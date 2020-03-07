# placeholder source file for Emscripten program wrapper
main.cpp:
	touch main.cpp

# Emscripten program wrapper
EXTRA_PROGRAMS = birch.js birch.html

birch_js_LDADD = libPACKAGE_TARNAME.la
birch_js_SOURCES = main.cpp
birch_js_LDFLAGS = -s MAIN_MODULE=1 --emrun

birch_html_LDADD = libPACKAGE_TARNAME.la
birch_html_SOURCES = main.cpp
birch_html_LDFLAGS = -s MAIN_MODULE=1 --emrun

AM_CPPFAGS = -DEIGEN_NO_STATIC_ASSERT -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1 
# -Deigen_assert=bi_assert
AM_CXXFLAGS = -include bi/PACKAGE_TARNAME.hpp $(OPENMP_CXXFLAGS
AM_CFLAGS = $(OPENMP_CFLAGS)

BUILT_SOURCES = main.cpp
CLEANFILES = main.cpp
