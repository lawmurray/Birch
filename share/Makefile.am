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

# compile and link flags
ACLOCAL_AMFLAGS = -I m4
AM_CPPFAGS = -DEIGEN_NO_STATIC_ASSERT -DEIGEN_NO_AUTOMATIC_RESIZING=1 -DEIGEN_DONT_PARALLELIZE=1 
# -Deigen_assert=bi_assert
AM_CXXFLAGS = -include bi/PACKAGE_TARNAME.hpp -fpic $(OPENMP_CXXFLAGS) -Wno-unknown-warning-option -Wno-overloaded-virtual -Wno-inconsistent-missing-override -Wno-return-type -Wno-defaulted-function-deleted
AM_CFLAGS = -fpic $(OPENMP_CFLAGS)

BUILT_SOURCES = main.cpp
CLEANFILES = main.cpp
