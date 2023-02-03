AC_INIT([MemBirch], [0.0.0], [], [membirch], [https://birch.sh])
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

# Programs
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_INSTALL
LT_INIT([dlopen disable-static])

# Checks for basic things
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
AC_HEADER_ASSERT
AC_OPENMP
CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"

if $checks; then
  # Checks for libraries
  AC_CHECK_LIB([atomic], [main], [], [], [])
  AS_IF([test "$OPENMP_CXXFLAGS" != ""], [AC_SEARCH_LIBS([omp_get_thread_num], [omp], [], [], [])])

  # Checks for headers
  AS_IF([test "$OPENMP_CXXFLAGS" != ""], [AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])])
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
