AC_INIT([PACKAGE_NAME], [0.0.0], [], [PACKAGE_TARNAME], [])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
LT_INIT([dlopen])
#AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXXCPP
AC_PROG_CXX
AC_PROG_INSTALL

AC_CHECK_DEFINE(__EMSCRIPTEN__, emscripten=true, emscripten=false)
if test x$emscripten = xtrue; then
  AC_DISABLE_SHARED
  AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
else
  AC_DISABLE_STATIC
  AX_CXX_COMPILE_STDCXX(14, [noext], [mandatory])
fi
AC_PROG_LIBTOOL

# Checks for basic things
AC_HEADER_ASSERT
AC_HEADER_STDBOOL
AC_C_INLINE

if test x$emscripten = xfalse; then
  # Checks for libraries
  AC_CHECK_LIB([birch], [main], [], [AC_MSG_ERROR([Birch compiler library not found])])
  AC_CHECK_LIB([ubjpp], [main], [], [AC_MSG_ERROR([Birch compiler library not found])])
  AC_CHECK_LIB([profiler], [main], [], [])
  if test xPACKAGE_TARNAME != xbirch_standard; then
      AC_CHECK_LIB([birch_standard], [main], [], [AC_MSG_ERROR([Birch standard library not found])])
  fi
  
  # Checks for headers
  AC_CHECK_HEADERS([libbirch/libbirch.hpp], [], [AC_MSG_ERROR([Birch compiler library header not found.])], [-])
  AC_CHECK_HEADERS([libubjpp/libubjpp.hpp], [], [AC_MSG_ERROR([Birch compiler library header not found.])], [-])
  if test xPACKAGE_TARNAME != xbirch_standard; then
      AC_CHECK_HEADERS([bi/birch_standard.hpp], [], [AC_MSG_ERROR([Birch standard library header not found.])], [-])
  fi
fi
