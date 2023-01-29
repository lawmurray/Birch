AC_INIT([{{PACKAGE_NAME}}], [{{PACKAGE_VERSION}}], [], [{{PACKAGE_TARNAME}}], [])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

AC_ARG_ENABLE([checks],
[AS_HELP_STRING[--enable-checks], [Enable header and library checks, can disable if just running make dist]],
[case "${enableval}" in
  yes) checks=true ;;
  no)  checks=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-checks]) ;;
esac],[checks=true])

AC_ARG_ENABLE([single],
[AS_HELP_STRING[--enable-single], [Build single precision libraries]],
[case "${enableval}" in
  yes) single=true ;;
  no)  single=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-single]) ;;
esac],[single=true])
AM_CONDITIONAL([SINGLE], [test x$single = xtrue])

AC_ARG_ENABLE([double],
[AS_HELP_STRING[--enable-double], [Build double precision libraries]],
[case "${enableval}" in
  yes) double=true ;;
  no)  double=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-double]) ;;
esac],[double=true])
AM_CONDITIONAL([DOUBLE], [test x$double = xtrue])

AC_ARG_ENABLE([standalone],
[AS_HELP_STRING[--enable-standalone], [Build standalone program]],
[case "${enableval}" in
  yes) standalone=true ;;
  no)  standalone=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-standalone]) ;;
esac],[standalone=false])
AM_CONDITIONAL([STANDALONE], [test x$standalone = xtrue])

# Programs
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_INSTALL

AC_DISABLE_STATIC
AC_ENABLE_SHARED
LT_INIT([dlopen,disable-static])

# Checks for basic things
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
AC_HEADER_ASSERT
AC_OPENMP
CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"

if $checks; then
  # Checks for compiler flags
  AX_CHECK_COMPILE_FLAG([-fno-caret-diagnostics], [CXXFLAGS="$CXXFLAGS -fno-caret-diagnostics"], [], [-Werror])
  AX_CHECK_COMPILE_FLAG([-fno-diagnostics-show-caret], [CXXFLAGS="$CXXFLAGS -fno-diagnostics-show-caret"], [], [-Werror])
  AX_CHECK_COMPILE_FLAG([-fno-diagnostics-show-option], [CXXFLAGS="$CXXFLAGS -fno-diagnostics-show-option"], [], [-Werror])
  AX_CHECK_COMPILE_FLAG([-fno-show-column], [CXXFLAGS="$CXXFLAGS -fno-show-column"], [], [-Werror])
  AX_CHECK_COMPILE_FLAG([-Wno-attributes], [CXXFLAGS="$CXXFLAGS -Wno-attributes"], [], [-Werror])
  AX_CHECK_COMPILE_FLAG([-Wno-int-to-pointer-cast], [CXXFLAGS="$CXXFLAGS  -Wno-int-to-pointer-cast"], [], [-Werror])
  AX_CHECK_COMPILE_FLAG([-Wno-unused-parameter], [CXXFLAGS="$CXXFLAGS -Wno-unused-parameter"], [], [-Werror])
  AX_CHECK_COMPILE_FLAG([-Wno-unknown-pragmas], [CXXFLAGS="$CXXFLAGS -Wno-unknown-pragmas"], [], [-Werror])

  AC_CHECK_LIB([membirch], [main], [], [AC_MSG_ERROR([required library not found])])
  if $single; then
    AC_CHECK_LIB([numbirch-single], [main], [SINGLE_LIBS="$SINGLE_LIBS -lnumbirch-single"], [AC_MSG_ERROR([required library not found])])
  fi
  if $double; then
    AC_CHECK_LIB([numbirch], [main], [DOUBLE_LIBS="$DOUBLE_LIBS -lnumbirch"], [AC_MSG_ERROR([required library not found])])
  fi
  if $standalone; then
    AC_CHECK_LIB([jemalloc], [mallocx], [STANDALONE_LIBS="$STANDALONE_LIBS -ljemalloc"], [])
  fi

  # Checks for libraries
  AC_CHECK_LIB([stdc++fs], [main], [], [])
  AS_IF([test "$OPENMP_CXXFLAGS" != ""], [AC_SEARCH_LIBS([omp_get_thread_num], [omp], [], [], [])])

  # Checks for headers
  AS_IF([test "$OPENMP_CXXFLAGS" != ""], [AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])])
  AC_CHECK_HEADERS([numbirch.hpp], [], [AC_MSG_ERROR([required header not found])], [AC_INCLUDES_DEFAULT])
  AC_CHECK_HEADERS([membirch.hpp], [], [AC_MSG_ERROR([required header not found])], [AC_INCLUDES_DEFAULT])
  AC_CHECK_HEADERS([filesystem], [], [], [AC_INCLUDES_DEFAULT])
  AC_CHECK_HEADERS([experimental/filesystem], [], [], [AC_INCLUDES_DEFAULT])
fi
