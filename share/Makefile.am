# precompiled header; note $(CXXCOMPILE) has a -include flag for the header,
# ensure that this won't use the old precompiled header by removing it first
bi/PACKAGE_TARNAME.hpp.pch: bi/PACKAGE_TARNAME.hpp
	rm -f $@
	mkdir -p bi/$(DEPDIR) && \
	$(CXXCOMPILE) -M -MF bi/$(DEPDIR)/PACKAGE_TARNAME.Plo -MT $@ $< && \
	$(CXXCOMPILE) -o $@ $<

# compile and link flags
ACLOCAL_AMFLAGS = -I m4
AM_CXXFLAGS = -include bi/PACKAGE_TARNAME.hpp

BUILT_SOURCES = bi/PACKAGE_TARNAME.hpp.pch
CLEANFILES = bi/PACKAGE_TARNAME.hpp.pch

include bi/$(DEPDIR)/PACKAGE_TARNAME.Plo
