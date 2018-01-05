AC_INIT([Birch], [0.0.0], [], [birch], [])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign])
#AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Checks for programs
AC_PROG_CPP
AC_PROG_CC
AC_PROG_CXXCPP
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_PROG_LEX
if test "x$LEX" = "x:"; then
  AC_MSG_ERROR([Flex not found.])
fi
AC_PROG_YACC
if test "x$YACC" = "xyacc"; then
  AC_MSG_ERROR([Bison not found.])
fi
AC_PATH_PROG([LEMON], [lemon], [])
if test "$LEMON" = ""; then
  AC_MSG_ERROR([lemon not found.])
fi
AX_CXX_COMPILE_STDCXX(14, [noext], [mandatory])

# Checks for libraries
AC_SEARCH_LIBS([dlopen], [dl], [], [])
AC_CHECK_LIB([m], [main], [], [AC_MSG_ERROR([required standard math library not found])])
AC_CHECK_LIB([boost_filesystem], [main], [], [AC_CHECK_LIB([boost_filesystem-mt], [main], [], [AC_MSG_ERROR([Boost.FileSystem library not found])])])
AC_CHECK_LIB([boost_system], [main], [], [AC_CHECK_LIB([boost_system-mt], [main], [], [AC_MSG_ERROR([Boost.System library not found])])])

# Checks for header files
AC_HEADER_ASSERT
    
AC_CHECK_HEADERS([ \
    boost/optional.hpp \
    boost/variant.hpp \
    boost/filesystem.hpp \
    boost/filesystem/fstream.hpp \
    boost/interprocess/sync/file_lock.hpp \
    boost/algorithm/string.hpp \
    ], [], AC_MSG_ERROR([required Boost header not found]), [-])

AC_CHECK_HEADERS([ \
    eigen3/Eigen/Core \
    eigen3/Eigen/Dense \
    eigen3/Eigen/Cholesky \
    ], [], AC_MSG_ERROR([required Eigen header not found]), [-])    

# Checks for typedefs, structures, and compiler characteristics
AC_HEADER_STDBOOL
AC_C_INLINE

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
