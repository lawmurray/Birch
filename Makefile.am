AUTOMAKE_OPTIONS = subdir-objects

# Flex/Bison setup
AM_YFLAGS = -d
AM_FLAGS =

# compiler setup
AM_CPPFLAGS = -DDATADIR="$(datadir)" -DINCLUDEDIR="$(includedir)" -DLIBDIR="$(libdir)"
AM_CXXFLAGS = -Wno-deprecated-register -Wno-overloaded-virtual
# ^ -Wno-deprecated-register to omit warnings for GNU Flex/Bison generated code
ACLOCAL_AMFLAGS = -I m4

# driver and compiler programs
bin_PROGRAMS = birch birchc

# compiler lirary
lib_LTLIBRARIES = libbirch.la

# common sources for driver and compiler programs
COMMON_SOURCES = \
  bi/build/Compiler.cpp \
  bi/build/Driver.cpp \
  bi/build/misc.cpp \
  bi/common/Argumented.cpp \
  bi/common/Based.cpp \
  bi/common/Braced.cpp \
  bi/common/Bracketed.cpp \
  bi/common/Conditioned.cpp \
  bi/common/Couple.cpp \
  bi/common/Dictionary.cpp \
  bi/common/Iterator.cpp \
  bi/common/List.cpp \
  bi/common/Located.cpp \
  bi/common/Location.cpp \
  bi/common/Name.cpp \
  bi/common/Named.cpp \
  bi/common/Numbered.cpp \
  bi/common/Overloaded.cpp \
  bi/common/OverloadedDictionary.cpp \
  bi/common/Parameterised.cpp \
  bi/common/Parenthesised.cpp \
  bi/common/Path.cpp \
  bi/common/ReturnTyped.cpp \
  bi/common/Reference.cpp \
  bi/common/Scope.cpp \
  bi/common/Scoped.cpp \
  bi/common/Single.cpp \
  bi/common/Typed.cpp \
  bi/common/Valued.cpp \
  bi/exception/AmbiguousCallException.cpp \
  bi/exception/CompilerException.cpp \
  bi/exception/CyclicImportException.cpp \
  bi/exception/DriverException.cpp \
  bi/exception/Exception.cpp \
  bi/exception/FileNotFoundException.cpp \
  bi/exception/GetException.cpp \
  bi/exception/InvalidAssignmentException.cpp \
  bi/exception/InvalidCallException.cpp \
  bi/exception/MemberException.cpp \
  bi/exception/NotAssignableException.cpp \
  bi/exception/NotFunctionException.cpp \
  bi/exception/PreviousDeclarationException.cpp \
  bi/exception/QueryException.cpp \
  bi/exception/SuperException.cpp \
  bi/exception/SuperBaseException.cpp \
  bi/exception/ThisException.cpp \
  bi/exception/UnresolvedException.cpp \
  bi/exception/UnsupportedOptionTypeException.cpp \
  bi/expression/Binary.cpp \
  bi/expression/BinaryCall.cpp \
  bi/expression/Brackets.cpp \
  bi/expression/Call.cpp \
  bi/expression/EmptyExpression.cpp \
  bi/expression/Expression.cpp \
  bi/expression/Get.cpp \
  bi/expression/Identifier.cpp \
  bi/expression/Index.cpp \
  bi/expression/LambdaFunction.cpp \
  bi/expression/Literal.cpp \
  bi/expression/Member.cpp \
  bi/expression/MemberParameter.cpp \
  bi/expression/Nil.cpp \
  bi/expression/OverloadedIdentifier.cpp \
  bi/expression/Parameter.cpp \
  bi/expression/Parentheses.cpp \
  bi/expression/Query.cpp \
  bi/expression/Range.cpp \
  bi/expression/Slice.cpp \
  bi/expression/Span.cpp \
  bi/expression/Super.cpp \
  bi/expression/This.cpp \
  bi/expression/UnaryCall.cpp \
  bi/io/cpp/CppBaseGenerator.cpp \
  bi/io/cpp/CppConstructorGenerator.cpp \
  bi/io/cpp/CppClassGenerator.cpp \
  bi/io/cpp/CppFiberGenerator.cpp \
  bi/io/cpp/CppForwardGenerator.cpp \
  bi/io/cpp/CppMemberFiberGenerator.cpp \
  bi/io/bi_ostream.cpp \
  bi/io/cpp_ostream.cpp \
  bi/io/indentable_ostream.cpp \
  bi/io/md_ostream.cpp \
  bi/primitive/encode.cpp \
  bi/statement/Alias.cpp \
  bi/statement/Assert.cpp \
  bi/statement/AssignmentOperator.cpp \
  bi/statement/Assignment.cpp \
  bi/statement/Basic.cpp \
  bi/statement/BinaryOperator.cpp \
  bi/statement/Class.cpp \
  bi/statement/ConversionOperator.cpp \
  bi/statement/EmptyStatement.cpp \
  bi/statement/ExpressionStatement.cpp \
  bi/statement/Fiber.cpp \
  bi/statement/File.cpp \
  bi/statement/For.cpp \
  bi/statement/Function.cpp \
  bi/statement/GlobalVariable.cpp \
  bi/statement/If.cpp \
  bi/statement/Import.cpp \
  bi/statement/LocalVariable.cpp \
  bi/statement/MemberFiber.cpp \
  bi/statement/MemberFunction.cpp \
  bi/statement/MemberVariable.cpp \
  bi/statement/Package.cpp \
  bi/statement/Program.cpp \
  bi/statement/Raw.cpp \
  bi/statement/Return.cpp \
  bi/statement/Statement.cpp \
  bi/statement/UnaryOperator.cpp \
  bi/statement/While.cpp \
  bi/statement/Yield.cpp \
  bi/type/AliasType.cpp \
  bi/type/ArrayType.cpp \
  bi/type/BasicType.cpp \
  bi/type/BinaryType.cpp \
  bi/type/ClassType.cpp \
  bi/type/EmptyType.cpp \
  bi/type/FiberType.cpp \
  bi/type/FunctionType.cpp \
  bi/type/IdentifierType.cpp \
  bi/type/ListType.cpp \
  bi/type/NilType.cpp \
  bi/type/OptionalType.cpp \
  bi/type/OverloadedType.cpp \
  bi/type/ParenthesesType.cpp \
  bi/type/Type.cpp \
  bi/visitor/Assigner.cpp \
  bi/visitor/Cloner.cpp \
  bi/visitor/Gatherer.cpp \
  bi/visitor/Modifier.cpp \
  bi/visitor/Resolver.cpp \
  bi/visitor/Typer.cpp \
  bi/visitor/Visitor.cpp \
  bi/lexer.lpp \
  bi/parser.ypp

# sources
birchc_LDADD = $(DEPS_LIBS)
birchc_SOURCES = \
  $(COMMON_SOURCES) \
  bi/birchc.cpp

birchc_CXXFLAGS = $(AM_CXXFLAGS)
# ^ workaround for 'created with both libtool and without' error

birch_LDADD = $(DEPS_LIBS)
birch_SOURCES = \
  $(COMMON_SOURCES) \
  bi/birch.cpp
birch_CXXFLAGS = $(AM_CXXFLAGS)

libbirch_la_SOURCES = \
  bi/lib/global.cpp \
  bi/libbirch.cpp
libbirch_la_CXXFLAGS = $(AM_CXXFLAGS)

nobase_include_HEADERS = \
  bi/lib/Array.hpp \
  bi/lib/Eigen.hpp \
  bi/lib/EigenFunctions.hpp \
  bi/lib/EigenOperators.hpp \
  bi/lib/Fiber.hpp \
  bi/lib/FiberState.hpp \
  bi/lib/Frame.hpp \
  bi/lib/global.hpp \
  bi/lib/Heap.hpp \
  bi/lib/Index.hpp \
  bi/lib/Iterator.hpp \
  bi/lib/Lead.hpp \
  bi/lib/Length.hpp \
  bi/lib/Object.hpp \
  bi/lib/Offset.hpp \
  bi/lib/Range.hpp \
  bi/lib/Pointer.hpp \
  bi/lib/Span.hpp \
  bi/lib/Stride.hpp \
  bi/lib/View.hpp \
  bi/libbirch.hpp
  
noinst_HEADERS = \
  bi/build/Compiler.hpp \
  bi/build/Driver.hpp \
  bi/build/misc.hpp \
  bi/common/Argumented.hpp \
  bi/common/Based.hpp \
  bi/common/Braced.hpp \
  bi/common/Bracketed.hpp \
  bi/common/Conditioned.hpp \
  bi/common/Couple.hpp \
  bi/common/Dictionary.hpp \
  bi/common/Iterator.hpp \
  bi/common/List.hpp \
  bi/common/Located.hpp \
  bi/common/Location.hpp \
  bi/common/Name.hpp \
  bi/common/Named.hpp \
  bi/common/Numbered.hpp \
  bi/common/Overloaded.hpp \
  bi/common/OverloadedDictionary.hpp \
  bi/common/Parameterised.hpp \
  bi/common/Parenthesised.hpp \
  bi/common/Path.hpp \
  bi/common/Reference.hpp \
  bi/common/ReturnTyped.hpp \
  bi/common/Scope.hpp \
  bi/common/Scoped.hpp \
  bi/common/Single.hpp \
  bi/common/Typed.hpp \
  bi/common/Valued.hpp \
  bi/exception/all.hpp \
  bi/exception/AmbiguousCallException.hpp \
  bi/exception/CompilerException.hpp \
  bi/exception/CyclicImportException.hpp \
  bi/exception/DriverException.hpp \
  bi/exception/FileNotFoundException.hpp \
  bi/exception/GetException.hpp \
  bi/exception/InvalidAssignmentException.hpp \
  bi/exception/InvalidCallException.hpp \
  bi/exception/MemberException.hpp \
  bi/exception/NotAssignableException.hpp \
  bi/exception/NotFunctionException.hpp \
  bi/exception/PreviousDeclarationException.hpp \
  bi/exception/QueryException.hpp \
  bi/exception/SuperBaseException.hpp \
  bi/exception/SuperException.hpp \
  bi/exception/ThisException.hpp \
  bi/exception/UnresolvedException.hpp \
  bi/exception/UnsupportedOptionTypeException.hpp \
  bi/expression/all.hpp \
  bi/expression/Binary.hpp \
  bi/expression/BinaryCall.hpp \
  bi/expression/Brackets.hpp \
  bi/expression/Call.hpp \
  bi/expression/EmptyExpression.hpp \
  bi/expression/Expression.hpp \
  bi/expression/Get.hpp \
  bi/expression/Identifier.hpp \
  bi/expression/Index.hpp \
  bi/expression/LambdaFunction.hpp \
  bi/expression/Literal.hpp \
  bi/expression/Member.hpp \
  bi/expression/MemberParameter.hpp \
  bi/expression/Nil.hpp \
  bi/expression/OverloadedIdentifier.hpp \
  bi/expression/Parameter.hpp \
  bi/expression/Parentheses.hpp \
  bi/expression/Query.hpp \
  bi/expression/Range.hpp \
  bi/expression/Slice.hpp \
  bi/expression/Span.hpp \
  bi/expression/Super.hpp \
  bi/expression/This.hpp \
  bi/expression/UnaryCall.hpp \
  bi/io/cpp/CppBaseGenerator.hpp \
  bi/io/cpp/CppConstructorGenerator.hpp \
  bi/io/cpp/CppClassGenerator.hpp \
  bi/io/cpp/CppFiberGenerator.hpp \
  bi/io/cpp/CppForwardGenerator.hpp \
  bi/io/cpp/CppMemberFiberGenerator.hpp \
  bi/io/cpp_ostream.hpp \
  bi/io/bi_ostream.hpp \
  bi/io/bih_ostream.hpp \
  bi/io/hpp_ostream.hpp \
  bi/io/indentable_ostream.hpp \
  bi/io/md_ostream.hpp \
  bi/primitive/definitely.hpp \
  bi/primitive/encode.hpp \
  bi/primitive/poset.hpp \
  bi/primitive/possibly.hpp \
  bi/statement/all.hpp \
  bi/statement/Alias.hpp \
  bi/statement/AssignmentOperator.hpp \
  bi/statement/Assignment.hpp \
  bi/statement/Basic.hpp \
  bi/statement/BinaryOperator.hpp \
  bi/statement/Class.hpp \
  bi/statement/ConversionOperator.hpp \
  bi/statement/EmptyStatement.hpp \
  bi/statement/ExpressionStatement.hpp \
  bi/statement/Fiber.hpp \
  bi/statement/File.hpp \
  bi/statement/For.hpp \
  bi/statement/Function.hpp \
  bi/statement/GlobalVariable.hpp \
  bi/statement/If.hpp \
  bi/statement/Import.hpp \
  bi/statement/LocalVariable.hpp \
  bi/statement/MemberFiber.hpp \
  bi/statement/MemberFunction.hpp \
  bi/statement/MemberVariable.hpp \
  bi/statement/Package.hpp \
  bi/statement/Program.hpp \
  bi/statement/Raw.hpp \
  bi/statement/Return.hpp \
  bi/statement/Statement.hpp \
  bi/statement/UnaryOperator.hpp \
  bi/statement/While.hpp \
  bi/statement/Yield.hpp \
  bi/type/all.hpp \
  bi/type/AliasType.hpp \
  bi/type/ArrayType.hpp \
  bi/type/BasicType.hpp \
  bi/type/BinaryType.hpp \
  bi/type/ClassType.hpp \
  bi/type/EmptyType.hpp \
  bi/type/FiberType.hpp \
  bi/type/FunctionType.hpp \
  bi/type/IdentifierType.hpp \
  bi/type/ListType.hpp \
  bi/type/NilType.hpp \
  bi/type/OptionalType.hpp \
  bi/type/OverloadedType.hpp \
  bi/type/ParenthesesType.hpp \
  bi/type/Type.hpp \
  bi/visitor/all.hpp \
  bi/visitor/Assigner.hpp \
  bi/visitor/Cloner.hpp \
  bi/visitor/Gatherer.hpp \
  bi/visitor/Modifier.hpp \
  bi/visitor/Resolver.hpp \
  bi/visitor/Visitor.hpp \
  bi/doxygen.hpp \
  bi/lexer.hpp \
  bi/parser.hpp

dist_noinst_SCRIPTS = autogen.sh
dist_pkgdata_DATA = \
  share/gitignore \
  share/ax_cxx_compile_stdcxx.m4 \
  share/ax_cxx_compile_stdcxx_11.m4 \
  share/ax_cxx_compile_stdcxx_14.m4 \
  share/LICENSE \
  share/MANIFEST \
  share/README.md \
  share/autogen.sh \
  share/common.am \
  share/configure.ac \
  share/precompile.hpp

# Automake docs suggest adding parser.hpp (#included in lexer.lpp but
# generated by bison running on parser.ypp) to BUILT_SOURCES to ensure that
# it exists before flex is run; this only works with `make all` and
# `make check` though.
BUILT_SOURCES = bi/parser.hpp
CLEANFILES = bi/parser.hpp
