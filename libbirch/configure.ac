AC_INIT([LibBirch], [unversioned], [], [libbirch], [https://birch.sh])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

# Both debug and release libraries are built, with AM_CXXFLAGS provided for
# these that should override the default CXXFLAGS; user-defined CXXFLAGS will
# still take precedence over these, however
: ${CXXFLAGS=""}

AC_ARG_ENABLE([debug],
[AS_HELP_STRING[--enable-debug], [Build debug library]],
[case "${enableval}" in
  yes) debug=true ;;
  no)  debug=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
esac],[debug=true])
AM_CONDITIONAL([DEBUG], [test x$debug = xtrue])

AC_ARG_ENABLE([test],
[AS_HELP_STRING[--enable-test], [Build test library]],
[case "${enableval}" in
  yes) test=true ;;
  no)  test=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-test]) ;;
esac],[test=false])
AM_CONDITIONAL([TEST], [test x$test = xtrue])

AC_ARG_ENABLE([release],
[AS_HELP_STRING[--enable-release], [Build release library]],
[case "${enableval}" in
  yes) release=true ;;
  no)  release=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-release]) ;;
esac],[release=true])
AM_CONDITIONAL([RELEASE], [test x$release = xtrue])

# Programs
AC_PROG_CXXCPP
AC_PROG_CXX
AC_PROG_INSTALL

AC_DISABLE_STATIC
AC_ENABLE_SHARED
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
LT_INIT([dlopen,pic-only])

# Checks for basic things
AC_HEADER_ASSERT
AC_HEADER_STDBOOL
AC_C_INLINE
AC_OPENMP

# Checks for compiler flags
AX_CHECK_COMPILE_FLAG([-fprofile-abs-path], [CXXFLAGS="$CXXFLAGS -fprofile-abs-path"], [], [-Werror])

# Checks for libraries
AC_SEARCH_LIBS([dlopen], [dl], [], [])
AC_CHECK_LIB([atomic], [main], [], [], [])
AC_CHECK_LIB([omp], [main], [], [], [])

# Checks for headers
AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])
AC_CHECK_HEADERS([eigen3/Eigen/Dense], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
