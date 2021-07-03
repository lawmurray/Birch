AC_INIT([PACKAGE_NAME], [PACKAGE_VERSION], [], [PACKAGE_TARNAME], [])
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
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
LT_INIT([dlopen,pic-only])

# Checks for basic things
AC_HEADER_ASSERT
AC_HEADER_STDBOOL
AC_OPENMP

# Checks for compiler flags
AX_CHECK_COMPILE_FLAG([-fno-caret-diagnostics], [CXXFLAGS="$CXXFLAGS -fno-caret-diagnostics"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-fno-diagnostics-show-caret], [CXXFLAGS="$CXXFLAGS -fno-diagnostics-show-caret"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-fno-diagnostics-show-option], [CXXFLAGS="$CXXFLAGS -fno-diagnostics-show-option"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-fno-show-column], [CXXFLAGS="$CXXFLAGS -fno-show-column"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-fprofile-abs-path], [CXXFLAGS="$CXXFLAGS -fprofile-abs-path"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-attributes], [CXXFLAGS="$CXXFLAGS -Wno-attributes"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-int-to-pointer-cast], [CXXFLAGS="$CXXFLAGS  -Wno-int-to-pointer-cast"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-unused-parameter], [CXXFLAGS="$CXXFLAGS -Wno-unused-parameter"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-unknown-pragmas], [CXXFLAGS="$CXXFLAGS -Wno-unknown-pragmas"], [], [-Werror])

# Checks for libraries
AC_CHECK_LIB([atomic], [main], [], [], [])
AC_CHECK_LIB([stdc++fs], [main], [], [], [])
AC_CHECK_LIB([omp], [main], [], [], [])

if $debug; then
  AC_CHECK_LIB([birch-debug], [main], [DEBUG_LIBS="$DEBUG_LIBS -lbirch-debug"], [AC_MSG_ERROR([required library not found])])
fi
if $test; then
  AC_CHECK_LIB([birch-test], [main], [TEST_LIBS="$TEST_LIBS -lbirch-test"], [AC_MSG_ERROR([required library not found])])
fi
if $release; then
  AC_CHECK_LIB([birch], [main], [RELEASE_LIBS="$RELEASE_LIBS -lbirch"], [AC_MSG_ERROR([required library not found])])
fi

# Checks for headers
AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])
AC_CHECK_HEADERS([libbirch.hpp], [], [AC_MSG_ERROR([required header not found])], [AC_INCLUDES_DEFAULT])
AC_CHECK_HEADERS([filesystem], [], [], [AC_INCLUDES_DEFAULT])
AC_CHECK_HEADERS([experimental/filesystem], [], [], [AC_INCLUDES_DEFAULT])
