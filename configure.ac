AC_INIT([Birch], [0.0.0], [], [birch], [])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign])
AC_CONFIG_MACRO_DIR([m4])

# Checks for programs
AC_PROG_CXXCPP
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LEX
if test "x$LEX" = "x:"; then
  AC_MSG_ERROR([Flex not found.])
fi
AC_PROG_YACC
if test "x$YACC" = "xyacc"; then
  AC_MSG_ERROR([Bison not found.])
fi

# Checks for basic things
AX_CXX_COMPILE_STDCXX(14, [noext], [mandatory])
AC_HEADER_ASSERT
AC_HEADER_STDBOOL

# Checks for headers
AC_CHECK_HEADERS([ \
    boost/filesystem.hpp \
    boost/filesystem/fstream.hpp \
    boost/property_tree/ptree.hpp \
    boost/property_tree/json_parser.hpp \
    boost/algorithm/string.hpp \
    ], [], AC_MSG_ERROR([required Boost header not found]), [-])
      
# Checks for libraries
AC_SEARCH_LIBS([dlopen], [dl], [], [])
AC_CHECK_LIB([boost_filesystem], [main], [], [AC_CHECK_LIB([boost_filesystem-mt], [main], [], [AC_MSG_ERROR([Boost.FileSystem library not found])])])
AC_CHECK_LIB([boost_system], [main], [], [AC_CHECK_LIB([boost_system-mt], [main], [], [AC_MSG_ERROR([Boost.System library not found])])])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
