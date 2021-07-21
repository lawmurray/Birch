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
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_INSTALL

AC_DISABLE_STATIC
AC_ENABLE_SHARED
LT_INIT([dlopen,pic-only])

# Checks for basic things
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
AC_HEADER_ASSERT
AC_OPENMP

# Checks for compiler flags
AX_CHECK_COMPILE_FLAG([-fprofile-abs-path], [CXXFLAGS="$CXXFLAGS -fprofile-abs-path"], [], [-Werror])

# Checks for libraries
AC_CHECK_LIB([atomic], [main], [], [], [])

if $debug; then
  AC_CHECK_LIB([numbirch-debug], [main], [DEBUG_LIBS="$DEBUG_LIBS -lnumbirch-debug"], [AC_MSG_ERROR([required library not found])])
fi
if $test; then
  AC_CHECK_LIB([numbirch-test], [main], [TEST_LIBS="$TEST_LIBS -lnumbirch-test"], [AC_MSG_ERROR([required library not found])])
fi
if $release; then
  AC_CHECK_LIB([numbirch], [main], [RELEASE_LIBS="$RELEASE_LIBS -lnumbirch"], [AC_MSG_ERROR([required library not found])])
fi

# Checks for headers
AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])
AC_CHECK_HEADERS([numbirch.hpp], [], [AC_MSG_ERROR([required header not found])], [AC_INCLUDES_DEFAULT])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
