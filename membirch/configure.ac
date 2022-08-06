AC_INIT([MemBirch], [unversioned], [], [membirch], [https://birch.sh])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

# Programs
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_INSTALL
LT_INIT([dlopen,disable-static])

# Checks for basic things
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
AC_HEADER_ASSERT
AC_OPENMP
CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"

# Checks for libraries
AC_CHECK_LIB([atomic], [main], [], [], [])
AS_IF([test "$OPENMP_CXXFLAGS" != ""], [AC_SEARCH_LIBS([omp_get_thread_num], [omp], [], [], [])])

# Checks for headers
AS_IF([test "$OPENMP_CXXFLAGS" != ""], [AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
