AC_INIT([Birch], [unversioned], [], [birch], [https://birch.sh])
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
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
AC_HEADER_ASSERT
AC_HEADER_STDBOOL

# Checks for compiler flags
AX_CHECK_COMPILE_FLAG([-fprofile-abs-path], [CXXFLAGS="$CXXFLAGS -fprofile-abs-path"], [], [-Werror])

# Checks for headers
AC_CHECK_HEADERS([yaml.h], [], [AC_MSG_ERROR([required header not found.])], [-])
AC_CHECK_HEADERS([glob.h], [], [AC_MSG_ERROR([required header not found.])], [-])
AC_CHECK_HEADERS([filesystem], [], [], [-])
AC_CHECK_HEADERS([experimental/filesystem], [], [], [-])

# Checks for libraries
AC_SEARCH_LIBS([dlopen], [dl], [], [])
AC_CHECK_LIB([stdc++fs], [main], [], [], [])
AC_CHECK_LIB([yaml], [main], [], [AC_MSG_ERROR([required library not found.])])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
