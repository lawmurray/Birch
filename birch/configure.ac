AC_INIT([Birch], [0.0.0], [], [birch], [https://birch.sh])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign])
AC_CONFIG_MACRO_DIR([m4])

AC_ARG_ENABLE([checks],
[AS_HELP_STRING[--enable-checks], [Enable header and library checks, can disable if just running make dist]],
[case "${enableval}" in
  yes) checks=true ;;
  no)  checks=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-checks]) ;;
esac],[checks=true])

# Checks for programs
AC_PROG_CXX
AC_PROG_CXXCPP
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

if $checks; then
  # Checks for headers
  AC_CHECK_HEADERS([yaml.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
  AC_CHECK_HEADERS([glob.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
  AC_CHECK_HEADERS([filesystem], [], [], [AC_INCLUDES_DEFAULT])
  AC_CHECK_HEADERS([experimental/filesystem], [], [], [AC_INCLUDES_DEFAULT])

  # Checks for libraries
  AC_CHECK_LIB([jemalloc], [malloc], [], [])
  AC_SEARCH_LIBS([dlopen], [dl], [], [])
  AC_CHECK_LIB([stdc++fs], [main], [], [])
  AC_CHECK_LIB([yaml], [main], [], [AC_MSG_ERROR([required library not found.])])
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
