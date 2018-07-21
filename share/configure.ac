AC_INIT([PACKAGE_NAME], [0.0.0], [], [PACKAGE_TARNAME], [])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CPP
AC_PROG_CC
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
LT_INIT([dlopen])

# Checks for basic things
AC_HEADER_ASSERT
AC_HEADER_STDBOOL
AC_C_INLINE
AC_OPENMP

# Checks for things that will fail under Emscripten, but won't prevent
# compilation
if test x$emscripten = xfalse; then  
  AC_SEARCH_LIBS([dlopen], [dl], [], [])
  AC_CHECK_LIB([boost_filesystem], [main], [], [AC_CHECK_LIB([boost_filesystem-mt], [main], [], [AC_MSG_ERROR([Boost.FileSystem library not found])])])
  AC_CHECK_LIB([boost_system], [main], [], [AC_CHECK_LIB([boost_system-mt], [main], [], [AC_MSG_ERROR([Boost.System library not found])])])

  # Checks for headers
  if test xPACKAGE_TARNAME != xbirch_standard; then
    AC_CHECK_HEADERS([bi/birch_standard.hpp], [], [AC_MSG_ERROR([Birch standard library header not found.])], [-])
  fi
fi

# Checks for libraries
if test xPACKAGE_TARNAME != xbirch_standard; then
  AC_CHECK_LIB([birch_standard], [main], [], [AC_MSG_ERROR([Birch standard library not found])])
fi

# Checks for functions
AX_GCC_BUILTIN([__builtin_clzll])
