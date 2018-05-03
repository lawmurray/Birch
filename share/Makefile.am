# precompiled header; note $(CXXCOMPILE) has a -include flag for the header,
# ensure that this won't use the old precompiled header by removing it first
bi/PACKAGE_TARNAME.hpp.pch: bi/PACKAGE_TARNAME.hpp
	rm -f $@
	mkdir -p bi/$(DEPDIR) && \
	$(CXXCOMPILE) -M -MF bi/$(DEPDIR)/PACKAGE_TARNAME.pch.Plo -MT $@ $< && \
	$(CXXCOMPILE) -o $@ $<

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
AM_CXXFLAGS = -include bi/PACKAGE_TARNAME.hpp

BUILT_SOURCES = bi/PACKAGE_TARNAME.hpp.pch main.cpp
CLEANFILES = bi/PACKAGE_TARNAME.hpp.pch main.cpp

include bi/$(DEPDIR)/PACKAGE_TARNAME.pch.Plo
