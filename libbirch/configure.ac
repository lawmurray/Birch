AC_INIT([LibBirch], [unversioned], [], [libbirch], [https://birch.sh])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

# Programs
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_INSTALL
LT_INIT([dlopen])

# Checks for basic things
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
AC_HEADER_ASSERT
AC_OPENMP

# Checks for libraries
AC_CHECK_LIB([atomic], [main], [], [], [])
AC_CHECK_LIB([omp], [main], [], [], [])

# Checks for headers
AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
