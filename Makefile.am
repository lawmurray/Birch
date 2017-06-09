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
  bi/common/FunctionMode.cpp \
  bi/common/Iterator.cpp \
  bi/common/List.cpp \
  bi/common/Located.cpp \
  bi/common/Location.cpp \
  bi/common/Name.cpp \
  bi/common/Named.cpp \
  bi/common/Numbered.cpp \
  bi/common/OverloadedDictionary.cpp \
  bi/common/OverloadedSet.cpp \
  bi/common/Parenthesised.cpp \
  bi/common/Path.cpp \
  bi/common/Reference.cpp \
  bi/common/Scope.cpp \
  bi/common/Scoped.cpp \
  bi/common/Signature.cpp \
  bi/common/Typed.cpp \
  bi/common/Unary.cpp \
  bi/common/VariableMode.cpp \
  bi/exception/AmbiguousReferenceException.cpp \
  bi/exception/CompilerException.cpp \
  bi/exception/CyclicImportException.cpp \
  bi/exception/Exception.cpp \
  bi/exception/FileNotFoundException.cpp \
  bi/exception/InvalidAssignmentException.cpp \
  bi/exception/MemberException.cpp \
  bi/exception/NotAssignableException.cpp \
  bi/exception/PreviousDeclarationException.cpp \
  bi/exception/SuperException.cpp \
  bi/exception/SuperBaseException.cpp \
  bi/exception/ThisException.cpp \
  bi/exception/UnresolvedReferenceException.cpp \
  bi/exception/UnsupportedOptionTypeException.cpp \
  bi/expression/BracesExpression.cpp \
  bi/expression/BracketsExpression.cpp \
  bi/expression/ConversionParameter.cpp \
  bi/expression/EmptyExpression.cpp \
  bi/expression/Expression.cpp \
  bi/expression/FuncParameter.cpp \
  bi/expression/FuncReference.cpp \
  bi/expression/Index.cpp \
  bi/expression/Literal.cpp \
  bi/expression/ParenthesesExpression.cpp \
  bi/expression/Range.cpp \
  bi/expression/Span.cpp \
  bi/expression/Super.cpp \
  bi/expression/This.cpp \
  bi/expression/Member.cpp \
  bi/expression/VarParameter.cpp \
  bi/expression/VarReference.cpp \
  bi/io/cpp/CppBaseGenerator.cpp \
  bi/io/cpp/CppConstructorGenerator.cpp \
  bi/io/cpp/CppCoroutineGenerator.cpp \
  bi/io/cpp/CppFileGenerator.cpp \
  bi/io/cpp/CppTypeGenerator.cpp \
  bi/io/cpp/CppParameterGenerator.cpp \
  bi/io/bi_ostream.cpp \
  bi/io/cpp_ostream.cpp \
  bi/io/indentable_ostream.cpp \
  bi/primitive/encode.cpp \
  bi/program/Prog.cpp \
  bi/program/ProgParameter.cpp \
  bi/program/ProgReference.cpp \
  bi/statement/If.cpp \
  bi/statement/Declaration.cpp \
  bi/statement/EmptyStatement.cpp \
  bi/statement/ExpressionStatement.cpp \
  bi/statement/File.cpp \
  bi/statement/For.cpp \
  bi/statement/Import.cpp \
  bi/statement/While.cpp \
  bi/statement/Raw.cpp \
  bi/statement/Return.cpp \
  bi/statement/Statement.cpp \
  bi/type/BracketsType.cpp \
  bi/type/EmptyType.cpp \
  bi/type/LambdaType.cpp \
  bi/type/TypeParameter.cpp \
  bi/type/TypeReference.cpp \
  bi/type/ParenthesesType.cpp \
  bi/type/Type.cpp \
  bi/visitor/Assigner.cpp \
  bi/visitor/Cloner.cpp \
  bi/visitor/Gatherer.cpp \
  bi/visitor/Modifier.cpp \
  bi/visitor/Resolver.cpp \
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

libbirch_la_SOURCES = bi/bi.cpp
libbirch_la_CXXFLAGS = $(AM_CXXFLAGS)

nobase_include_HEADERS = \
  bi/lib/Array.hpp \
  bi/lib/constant.hpp \
  bi/lib/easy.hpp \
  bi/lib/Frame.hpp \
  bi/lib/memory.hpp \
  bi/lib/Index.hpp \
  bi/lib/Iterator.hpp \
  bi/lib/Lead.hpp \
  bi/lib/Length.hpp \
  bi/lib/Offset.hpp \
  bi/lib/Range.hpp \
  bi/lib/reloc_ptr.hpp \
  bi/lib/Span.hpp \
  bi/lib/Stride.hpp \
  bi/lib/View.hpp \
  bi/bi.hpp
  
noinst_HEADERS = \
  bi/build/Compiler.hpp \
  bi/build/Driver.hpp \
  bi/build/misc.hpp \
  bi/build/Packager.hpp \
  bi/common/Based.hpp \
  bi/common/Binary.hpp \
  bi/common/Braced.hpp \
  bi/common/Bracketed.hpp \
  bi/common/Conditioned.hpp \
  bi/common/Dictionary.hpp \
  bi/common/FunctionMode.hpp \
  bi/common/Iterator.hpp \
  bi/common/List.hpp \
  bi/common/Located.hpp \
  bi/common/Location.hpp \
  bi/common/Name.hpp \
  bi/common/Named.hpp \
  bi/common/Numbered.hpp \
  bi/common/OverloadedDictionary.hpp \
  bi/common/OverloadedSet.hpp \
  bi/common/Parenthesised.hpp \
  bi/common/Path.hpp \
  bi/common/Reference.hpp \
  bi/common/Scope.hpp \
  bi/common/Scoped.hpp \
  bi/common/Signature.hpp \
  bi/common/Typed.hpp \
  bi/common/Unary.hpp \
  bi/common/VariableMode.hpp \
  bi/config.hpp \
  bi/exception/all.hpp \
  bi/exception/AmbiguousReferenceException.hpp \
  bi/exception/CompilerException.hpp \
  bi/exception/CyclicImportException.hpp \
  bi/exception/DriverException.hpp \
  bi/exception/FileNotFoundException.hpp \
  bi/exception/InvalidAssignmentException.hpp \
  bi/exception/MemberException.hpp \
  bi/exception/NotAssignableException.hpp \
  bi/exception/PreviousDeclarationException.hpp \
  bi/exception/SuperBaseException.hpp \
  bi/exception/SuperException.hpp \
  bi/exception/ThisException.hpp \
  bi/exception/UnresolvedReferenceException.hpp \
  bi/exception/UnsupportedOptionTypeException.hpp \
  bi/expression/all.hpp \
  bi/expression/BracesExpression.hpp \
  bi/expression/BracketsExpression.hpp \
  bi/expression/ConversionParameter.hpp \
  bi/expression/EmptyExpression.hpp \
  bi/expression/Expression.hpp \
  bi/expression/FuncParameter.hpp \
  bi/expression/FuncReference.hpp \
  bi/expression/Index.hpp \
  bi/expression/Literal.hpp \
  bi/expression/Member.hpp \
  bi/expression/ParenthesesExpression.hpp \
  bi/expression/Range.hpp \
  bi/expression/Span.hpp \
  bi/expression/Super.hpp \
  bi/expression/This.hpp \
  bi/expression/VarParameter.hpp \
  bi/expression/VarReference.hpp \
  bi/io/bi_ostream.hpp \
  bi/io/bih_ostream.hpp \
  bi/io/cpp/CppBaseGenerator.hpp \
  bi/io/cpp/CppConstructorGenerator.hpp \
  bi/io/cpp/CppCoroutineGenerator.hpp \
  bi/io/cpp/CppFileGenerator.hpp \
  bi/io/cpp/CppParameterGenerator.hpp \
  bi/io/cpp/CppTypeGenerator.hpp \
  bi/io/cpp_ostream.hpp \
  bi/io/hpp_ostream.hpp \
  bi/io/indentable_ostream.hpp \
  bi/lexer.hpp \
  bi/modules.hpp \
  bi/parser.hpp \
  bi/primitive/definitely.hpp \
  bi/primitive/encode.hpp \
  bi/primitive/poset.hpp \
  bi/primitive/possibly.hpp \
  bi/primitive/shared_ptr.hpp \
  bi/primitive/unique_ptr.hpp \
  bi/program/all.hpp \
  bi/program/Prog.hpp \
  bi/program/ProgParameter.hpp \
  bi/program/ProgReference.hpp \
  bi/statement/all.hpp \
  bi/statement/Declaration.hpp \
  bi/statement/EmptyStatement.hpp \
  bi/statement/ExpressionStatement.hpp \
  bi/statement/File.hpp \
  bi/statement/For.hpp \
  bi/statement/If.hpp \
  bi/statement/Import.hpp \
  bi/statement/Raw.hpp \
  bi/statement/Return.hpp \
  bi/statement/Statement.hpp \
  bi/statement/While.hpp \
  bi/type/all.hpp \
  bi/type/BracketsType.hpp \
  bi/type/EmptyType.hpp \
  bi/type/LambdaType.hpp \
  bi/type/ParenthesesType.hpp \
  bi/type/Type.hpp \
  bi/type/TypeParameter.hpp \
  bi/type/TypeReference.hpp \
  bi/visitor/all.hpp \
  bi/visitor/Assigner.hpp \
  bi/visitor/Cloner.hpp \
  bi/visitor/Gatherer.hpp \
  bi/visitor/Modifier.hpp \
  bi/visitor/Resolver.hpp \
  bi/visitor/Visitor.hpp

dist_noinst_SCRIPTS = autogen.sh
dist_pkgdata_DATA = \
  share/gitignore \
  share/ax_cxx_compile_stdcxx.m4 \
  share/ax_cxx_compile_stdcxx_11.m4 \
  share/ax_cxx_compile_stdcxx_14.m4 \
  share/LICENSE \
  share/Makefile \
  share/MANIFEST \
  share/META.md \
  share/README.md \
  share/VERSION.md \
  share/autogen.sh \
  share/common.am \
  share/configure.ac

# Automake docs suggest adding parser.hpp (#included in lexer.lpp but
# generated by bison running on parser.ypp) to BUILT_SOURCES to ensure that
# it exists before flex is run; this only works with `make all` and
# `make check` though.
BUILT_SOURCES = bi/parser.hpp
