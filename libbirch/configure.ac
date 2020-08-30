AC_INIT([LibBirch], [unversioned], [], [libbirch], [https://birch-lang.org])
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
esac],[release=false])
AM_CONDITIONAL([RELEASE], [test x$release = xtrue])

# Programs
AC_PROG_CXXCPP
AC_PROG_CXX
AC_PROG_INSTALL

AC_CHECK_DEFINE(__EMSCRIPTEN__, emscripten=true, emscripten=false)
if test x$emscripten = xtrue; then
  AC_ENABLE_STATIC
  AC_DISABLE_SHARED
  AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
else
  AC_DISABLE_STATIC
  AC_ENABLE_SHARED
  AX_CXX_COMPILE_STDCXX(14, [noext], [mandatory])
fi
LT_INIT([dlopen,pic-only])

# Checks for basic things
AC_HEADER_ASSERT
AC_HEADER_STDBOOL
AC_C_INLINE
AC_OPENMP

# Checks for functions
AX_GCC_BUILTIN([__builtin_clz])
AX_GCC_BUILTIN([__builtin_clzll])

# Checks for things that fail under Emscripten, but won't prevent compilation
if test x$emscripten = xfalse; then
  AC_SEARCH_LIBS([dlopen], [dl], [], [])
  AC_CHECK_LIB([atomic], [main], [], [], [])
  AC_CHECK_LIB([omp], [main], [], [], [])

  # Checks for headers
  AC_CHECK_HEADERS([omp.h], [], [], [-])
  AC_CHECK_HEADERS([eigen3/Eigen/Dense], [], [AC_MSG_ERROR([required header not found.])], [-])
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
