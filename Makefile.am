AUTOMAKE_OPTIONS = subdir-objects

# Flex/Bison setup
AM_YFLAGS = -d
AM_FLAGS =

# compiler setup
AM_CPPFLAGS = -DDATADIR="$(datadir)" -DINCLUDEDIR="$(includedir)" -DLIBDIR="$(libdir)"
AM_CXXFLAGS = -Wno-deprecated-register -Wno-overloaded-virtual
# ^ -Wno-deprecated-register to omit warnings for GNU Flex/Bison generated code
ACLOCAL_AMFLAGS = -I m4

# programs
bin_PROGRAMS = birchc birch

# libraries
lib_LTLIBRARIES = libbirch.la

# sources
birchc_LDADD = $(DEPS_LIBS)
birchc_SOURCES = \
  bi/build/Compiler.cpp \
  bi/build/misc.cpp \
  bi/common/Based.cpp \
  bi/common/Binary.cpp \
  bi/common/Braced.cpp \
  bi/common/Bracketed.cpp \
  bi/common/Conditioned.cpp \
  bi/common/Dictionary.cpp \
  bi/common/Formed.cpp \
  bi/common/List.cpp \
  bi/common/Located.cpp \
  bi/common/Location.cpp \
  bi/common/Name.cpp \
  bi/common/Named.cpp \
  bi/common/Numbered.cpp \
  bi/common/OverloadedDictionary.cpp \
  bi/common/Parameter.cpp \
  bi/common/Parenthesised.cpp \
  bi/common/Path.cpp \
  bi/common/Reference.cpp \
  bi/common/Scope.cpp \
  bi/common/Scoped.cpp \
  bi/common/Typed.cpp \
  bi/common/Unary.cpp \
  bi/exception/AmbiguousReferenceException.cpp \
  bi/exception/CompilerException.cpp \
  bi/exception/CyclicImportException.cpp \
  bi/exception/Exception.cpp \
  bi/exception/FileNotFoundException.cpp \
  bi/exception/MemberException.cpp \
  bi/exception/NotAssignableException.cpp \
  bi/exception/PreviousDeclarationException.cpp \
  bi/exception/ThisException.cpp \
  bi/exception/UnresolvedReferenceException.cpp \
  bi/exception/UnsupportedOptionTypeException.cpp \
  bi/expression/BracesExpression.cpp \
  bi/expression/BracketsExpression.cpp \
  bi/expression/EmptyExpression.cpp \
  bi/expression/Expression.cpp \
  bi/expression/FuncParameter.cpp \
  bi/expression/FuncReference.cpp \
  bi/expression/Index.cpp \
  bi/expression/Literal.cpp \
  bi/expression/ParenthesesExpression.cpp \
  bi/expression/RandomInit.cpp \
  bi/expression/Range.cpp \
  bi/expression/This.cpp \
  bi/expression/Member.cpp \
  bi/expression/VarParameter.cpp \
  bi/expression/VarReference.cpp \
  bi/io/cpp/CppAssignmentGenerator.cpp \
  bi/io/cpp/CppBaseGenerator.cpp \
  bi/io/cpp/CppConstructorGenerator.cpp \
  bi/io/cpp/CppCopyConstructorGenerator.cpp \
  bi/io/cpp/CppDispatcherGenerator.cpp \
  bi/io/cpp/CppFileGenerator.cpp \
  bi/io/cpp/CppModelGenerator.cpp \
  bi/io/cpp/CppOutputGenerator.cpp \
  bi/io/cpp/CppParameterGenerator.cpp \
  bi/io/cpp/CppReturnGenerator.cpp \
  bi/io/cpp/CppViewConstructorGenerator.cpp \
  bi/io/cpp/misc.cpp \
  bi/io/bi_ostream.cpp \
  bi/io/cpp_ostream.cpp \
  bi/io/indentable_ostream.cpp \
  bi/io/mangler_ostream.cpp \
  bi/primitive/encode.cpp \
  bi/primitive/possibly.cpp \
  bi/primitive/signature_less_equal.cpp \
  bi/program/Prog.cpp \
  bi/program/ProgParameter.cpp \
  bi/program/ProgReference.cpp \
  bi/statement/Conditional.cpp \
  bi/statement/Declaration.cpp \
  bi/statement/EmptyStatement.cpp \
  bi/statement/ExpressionStatement.cpp \
  bi/statement/File.cpp \
  bi/statement/Import.cpp \
  bi/statement/Loop.cpp \
  bi/statement/Raw.cpp \
  bi/statement/Statement.cpp \
  bi/type/AssignableType.cpp \
  bi/type/BracketsType.cpp \
  bi/type/EmptyType.cpp \
  bi/type/ModelParameter.cpp \
  bi/type/ModelReference.cpp \
  bi/type/ParenthesesType.cpp \
  bi/type/RandomType.cpp \
  bi/type/Type.cpp \
  bi/visitor/Assigner.cpp \
  bi/visitor/Cloner.cpp \
  bi/visitor/DispatchGatherer.cpp \
  bi/visitor/Gatherer.cpp \
  bi/visitor/IsPrimary.cpp \
  bi/visitor/IsRandom.cpp \
  bi/visitor/IsRich.cpp \
  bi/visitor/Modifier.cpp \
  bi/visitor/Resolver.cpp \
  bi/visitor/TupleSizer.cpp \
  bi/visitor/Visitor.cpp \
  bi/birchc.cpp \
  bi/lexer.lpp \
  bi/parser.ypp
birchc_CXXFLAGS = $(AM_CXXFLAGS)
# ^ workaround for 'created with both libtool and without' error

birch_LDADD = $(DEPS_LIBS)
birch_SOURCES = \
  bi/build/Driver.cpp \
  bi/build/misc.cpp \
  bi/build/Packager.cpp \
  bi/exception/DriverException.cpp \
  bi/exception/Exception.cpp \
  bi/exception/FileNotFoundException.cpp \
  bi/birch.cpp \
  bi/config.lpp
birch_CXXFLAGS = $(AM_CXXFLAGS)

libbirch_la_SOURCES = \
  bi/bi.cpp \
  bi/data/copy.cpp \
  bi/data/group.cpp \
  bi/data/netcdf.cpp \
  bi/data/MemoryGroup.cpp \
  bi/data/MemoryPrimitiveValue.cpp \
  bi/data/NetCDFGroup.cpp \
  bi/data/NetCDFPrimitiveValue.cpp \
  bi/data/NetCDFView.cpp \
  bi/exception/Exception.cpp \
  bi/exception/MemoryException.cpp \
  bi/exception/NetCDFException.cpp \
  bi/exception/UnknownOptionException.cpp \
  bi/random/RandomStack.cpp
libbirch_la_CXXFLAGS = $(AM_CXXFLAGS)

nobase_include_HEADERS = bi/data/Array.hpp \
  bi/data/builtin.hpp \
  bi/data/constant.hpp \
  bi/data/copy.hpp \
  bi/data/easy.hpp \
  bi/data/group.hpp \
  bi/data/Frame.hpp \
  bi/data/MemoryGroup.hpp \
  bi/data/MemoryPrimitiveValue.hpp \
  bi/data/Index.hpp \
  bi/data/Iterator.hpp \
  bi/data/Lead.hpp \
  bi/data/Length.hpp \
  bi/data/netcdf.hpp \
  bi/data/NetCDFGroup.hpp \
  bi/data/NetCDFPrimitiveValue.hpp \
  bi/data/NetCDFView.hpp \
  bi/data/Offset.hpp \
  bi/data/PrimitiveValue.hpp \
  bi/data/Range.hpp \
  bi/data/Span.hpp \
  bi/data/Stride.hpp \
  bi/data/View.hpp \
  bi/exception/Exception.hpp \
  bi/exception/MemoryException.hpp \
  bi/exception/NetCDFException.hpp \
  bi/exception/UnknownOptionException.hpp \
  bi/primitive/cast.hpp \
  bi/primitive/shared_ptr.hpp \
  bi/primitive/unique_ptr.hpp \
  bi/random/Random.hpp \
  bi/random/RandomStack.hpp \
  bi/bi.hpp

dist_noinst_SCRIPTS = autogen.sh
dist_pkgdata_DATA = \
  share/gitignore \
  share/ax_cxx_compile_stdcxx.m4 \
  share/ax_cxx_compile_stdcxx_11.m4 \
  share/ax_cxx_compile_stdcxx_14.m4 \
  share/LICENSE \
  share/Makefile \
  share/MANIFEST \
  share/autogen.sh \
  share/common.am \
  share/configure.ac

# Automake docs suggest adding parser.hpp (#included in lexer.lpp but
# generated by bison running on parser.ypp) to BUILT_SOURCES to ensure that
# it exists before flex is run; this only works with `make all` and
# `make check` though.
BUILT_SOURCES = bi/parser.hpp bi/lexer.hpp bi/config.hpp
