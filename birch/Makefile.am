AUTOMAKE_OPTIONS = subdir-objects

# Flex/Bison setup
AM_YFLAGS = -d -Wno-yacc
AM_FLAGS =

# compiler setup
AM_CPPFLAGS = -DPREFIX="$(prefix)" -DDATADIR="$(datadir)" -DINCLUDEDIR="$(includedir)" -DLIBDIR="$(libdir)"
AM_CXXFLAGS = -include src/birch.hpp

# driver program
bin_PROGRAMS = birch

# sources
birch_SOURCES = \
  src/build/Compiler.cpp \
  src/build/Driver.cpp \
  src/build/MetaParser.cpp \
  src/common/Annotated.cpp \
  src/common/Argumented.cpp \
  src/common/Based.cpp \
  src/common/Braced.cpp \
  src/common/Bracketed.cpp \
  src/common/Conditioned.cpp \
  src/common/Couple.cpp \
  src/common/Located.cpp \
  src/common/Location.cpp \
  src/common/Name.cpp \
  src/common/Named.cpp \
  src/common/Numbered.cpp \
  src/common/Parameterised.cpp \
  src/common/ReturnTyped.cpp \
  src/common/Single.cpp \
  src/common/Typed.cpp \
  src/common/TypeArgumented.cpp \
  src/common/TypeParameterised.cpp \
  src/common/Valued.cpp \
  src/exception/DriverException.cpp \
  src/exception/Exception.cpp \
  src/exception/FileNotFoundException.cpp \
  src/expression/Assign.cpp \
  src/expression/BinaryCall.cpp \
  src/expression/Cast.cpp \
  src/expression/Call.cpp \
  src/expression/EmptyExpression.cpp \
  src/expression/Expression.cpp \
  src/expression/ExpressionConstIterator.cpp \
  src/expression/ExpressionIterator.cpp \
  src/expression/ExpressionList.cpp \
  src/expression/Generic.cpp \
  src/expression/Get.cpp \
  src/expression/Global.cpp \
  src/expression/LambdaFunction.cpp \
  src/expression/Literal.cpp \
  src/expression/Member.cpp \
  src/expression/NamedExpression.cpp \
  src/expression/Nil.cpp \
  src/expression/Parameter.cpp \
  src/expression/Parentheses.cpp \
  src/expression/Query.cpp \
  src/expression/Range.cpp \
  src/expression/Sequence.cpp \
  src/expression/Slice.cpp \
  src/expression/Span.cpp \
  src/expression/Super.cpp \
  src/expression/This.cpp \
  src/expression/UnaryCall.cpp \
  src/generate/BirchGenerator.cpp \
  src/generate/CppClassGenerator.cpp \
  src/generate/CppGenerator.cpp \
  src/generate/CppPackageGenerator.cpp \
  src/generate/CppStructGenerator.cpp \
  src/generate/IndentableGenerator.cpp \
  src/generate/MarkdownGenerator.cpp \
  src/primitive/string.cpp \
  src/primitive/system.cpp \
  src/statement/Assert.cpp \
  src/statement/AssignmentOperator.cpp \
  src/statement/Basic.cpp \
  src/statement/BinaryOperator.cpp \
  src/statement/Block.cpp \
  src/statement/Braces.cpp \
  src/statement/Class.cpp \
  src/statement/ConversionOperator.cpp \
  src/statement/DoWhile.cpp \
  src/statement/EmptyStatement.cpp \
  src/statement/ExpressionStatement.cpp \
  src/statement/Factor.cpp \
  src/statement/File.cpp \
  src/statement/For.cpp \
  src/statement/Function.cpp \
  src/statement/GlobalVariable.cpp \
  src/statement/If.cpp \
  src/statement/LocalVariable.cpp \
  src/statement/MemberFunction.cpp \
  src/statement/MemberPhantom.cpp \
  src/statement/MemberVariable.cpp \
  src/statement/Package.cpp \
  src/statement/Parallel.cpp \
  src/statement/Program.cpp \
  src/statement/Raw.cpp \
  src/statement/Return.cpp \
  src/statement/SliceOperator.cpp \
  src/statement/Statement.cpp \
  src/statement/StatementIterator.cpp \
  src/statement/StatementList.cpp \
  src/statement/Struct.cpp \
  src/statement/TupleVariable.cpp \
  src/statement/UnaryOperator.cpp \
  src/statement/While.cpp \
  src/statement/With.cpp \
  src/type/ArrayType.cpp \
  src/type/DeducedType.cpp \
  src/type/EmptyType.cpp \
  src/type/FutureType.cpp \
  src/type/MemberType.cpp \
  src/type/NamedType.cpp \
  src/type/OptionalType.cpp \
  src/type/TupleType.cpp \
  src/type/Type.cpp \
  src/type/TypeConstIterator.cpp \
  src/type/TypeIterator.cpp \
  src/type/TypeList.cpp \
  src/visitor/Gatherer.cpp \
  src/visitor/Visitor.cpp \
  src/birch.cpp \
  src/lexer.lpp \
  src/parser.ypp

birch_CXXFLAGS = $(AM_CXXFLAGS)

noinst_HEADERS = \
  src/build/Compiler.hpp \
  src/build/Driver.hpp \
  src/build/MetaParser.hpp \
  src/common/Annotated.hpp \
  src/common/Argumented.hpp \
  src/common/Based.hpp \
  src/common/Braced.hpp \
  src/common/Bracketed.hpp \
  src/common/Conditioned.hpp \
  src/common/Couple.hpp \
  src/common/Located.hpp \
  src/common/Location.hpp \
  src/common/Name.hpp \
  src/common/Named.hpp \
  src/common/Numbered.hpp \
  src/common/Parameterised.hpp \
  src/common/ReturnTyped.hpp \
  src/common/Single.hpp \
  src/common/Typed.hpp \
  src/common/TypeArgumented.hpp \
  src/common/TypeParameterised.hpp \
  src/common/Valued.hpp \
  src/exception/all.hpp \
  src/exception/DriverException.hpp \
  src/exception/Exception.hpp \
  src/exception/FileNotFoundException.hpp \
  src/expression/all.hpp \
  src/expression/Assign.hpp \
  src/expression/BinaryCall.hpp \
  src/expression/Cast.hpp \
  src/expression/Call.hpp \
  src/expression/EmptyExpression.hpp \
  src/expression/Expression.hpp \
  src/expression/ExpressionConstIterator.hpp \
  src/expression/ExpressionIterator.hpp \
  src/expression/ExpressionList.hpp \
  src/expression/Generic.hpp \
  src/expression/Get.hpp \
  src/expression/Global.hpp \
  src/expression/LambdaFunction.hpp \
  src/expression/Literal.hpp \
  src/expression/Member.hpp \
  src/expression/NamedExpression.hpp \
  src/expression/Nil.hpp \
  src/expression/Parameter.hpp \
  src/expression/Parentheses.hpp \
  src/expression/Query.hpp \
  src/expression/Range.hpp \
  src/expression/Sequence.hpp \
  src/expression/Slice.hpp \
  src/expression/Span.hpp \
  src/expression/Super.hpp \
  src/expression/This.hpp \
  src/expression/UnaryCall.hpp \
  src/generate/BirchGenerator.hpp \
  src/generate/CppClassGenerator.hpp \
  src/generate/CppGenerator.hpp \
  src/generate/CppPackageGenerator.hpp \
  src/generate/CppStructGenerator.hpp \
  src/generate/IndentableGenerator.hpp \
  src/generate/MarkdownGenerator.hpp \
  src/primitive/inherits.hpp \
  src/primitive/poset.hpp \
  src/primitive/string.hpp \
  src/primitive/system.hpp \
  src/statement/all.hpp \
  src/statement/AssignmentOperator.hpp \
  src/statement/Assert.hpp \
  src/statement/Basic.hpp \
  src/statement/BinaryOperator.hpp \
  src/statement/Block.hpp \
  src/statement/Braces.hpp \
  src/statement/Class.hpp \
  src/statement/ConversionOperator.hpp \
  src/statement/DoWhile.hpp \
  src/statement/EmptyStatement.hpp \
  src/statement/ExpressionStatement.hpp \
  src/statement/Factor.hpp \
  src/statement/File.hpp \
  src/statement/For.hpp \
  src/statement/Function.hpp \
  src/statement/GlobalVariable.hpp \
  src/statement/If.hpp \
  src/statement/LocalVariable.hpp \
  src/statement/MemberFunction.hpp \
  src/statement/MemberPhantom.hpp \
  src/statement/MemberVariable.hpp \
  src/statement/Package.hpp \
  src/statement/Parallel.hpp \
  src/statement/Program.hpp \
  src/statement/Raw.hpp \
  src/statement/Return.hpp \
  src/statement/SliceOperator.hpp \
  src/statement/Statement.hpp \
  src/statement/StatementIterator.hpp \
  src/statement/StatementList.hpp \
  src/statement/Struct.hpp \
  src/statement/TupleVariable.hpp \
  src/statement/UnaryOperator.hpp \
  src/statement/While.hpp \
  src/statement/With.hpp \
  src/type/all.hpp \
  src/type/ArrayType.hpp \
  src/type/DeducedType.hpp \
  src/type/EmptyType.hpp \
  src/type/FutureType.hpp \
  src/type/MemberType.hpp \
  src/type/NamedType.hpp \
  src/type/OptionalType.hpp \
  src/type/TupleType.hpp \
  src/type/Type.hpp \
  src/type/TypeConstIterator.hpp \
  src/type/TypeIterator.hpp \
  src/type/TypeList.hpp \
  src/visitor/all.hpp \
  src/visitor/Gatherer.hpp \
  src/visitor/Visitor.hpp \
  src/birch.hpp \
  src/doxygen.hpp \
  src/lexer.hpp

dist_pkgdata_DATA = \
  share/gitignore \
  share/ax_check_compile_flag.m4 \
  share/ax_check_define.m4 \
  share/ax_cxx_compile_stdcxx.m4 \
  share/ax_gcc_builtin.m4 \
  share/birch.yml \
  share/bootstrap \
  share/configure.ac \
  share/hello.birch \
  share/LICENSE \
  share/Makefile.am \
	share/mkdocs.yml  \
  share/README.md \
  share/standalone.cpp

dist_noinst_DATA = \
  Doxyfile \
  LICENSE \
  README.md

BUILT_SOURCES = src/parser.hpp src/parser.cpp src/lexer.cpp
CLEANFILES = $(BUILT_SOURCES)
