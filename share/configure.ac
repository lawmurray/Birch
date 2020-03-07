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
LT_INIT([dlopen,pic-only])

# Checks for basic things
AC_HEADER_ASSERT
AC_HEADER_STDBOOL
AC_C_INLINE
AC_OPENMP

# Checks for compiler flags
AX_CHECK_COMPILE_FLAG([-fno-caret-diagnostics], [CXXFLAGS="$CXXFLAGS -fno-caret-diagnostics"])
AX_CHECK_COMPILE_FLAG([-fno-diagnostics-show-option], [CXXFLAGS="$CXXFLAGS -fno-diagnostics-show-option"])
AX_CHECK_COMPILE_FLAG([-Wno-overloaded-virtual], [CXXFLAGS="$CXXFLAGS -Wno-overloaded-virtual"])
AX_CHECK_COMPILE_FLAG([-Wno-inconsistent-missing-override], [CXXFLAGS="$CXXFLAGS -Wno-inconsistent-missing-override"])
AX_CHECK_COMPILE_FLAG([-Wno-attributes], [CXXFLAGS="$CXXFLAGS -Wno-attributes"])
AX_CHECK_COMPILE_FLAG([-Wno-int-to-pointer-cast], [CXXFLAGS="$CXXFLAGS  -Wno-int-to-pointer-cast"])
AX_CHECK_COMPILE_FLAG([-Wno-unused-parameter], [CXXFLAGS="$CXXFLAGS -Wno-unused-parameter"])
AX_CHECK_COMPILE_FLAG([-Wno-unused-variable], [CXXFLAGS="$CXXFLAGS -Wno-unused-variable"])
AX_CHECK_COMPILE_FLAG([-Wno-unused-value], [CXXFLAGS="$CXXFLAGS -Wno-unused-value"])
AX_CHECK_COMPILE_FLAG([-Wno-unused-local-typedefs], [CXXFLAGS="$CXXFLAGS -Wno-unused-local-typedefs"])
AX_CHECK_COMPILE_FLAG([-Wno-unknown-pragmas], [CXXFLAGS="$CXXFLAGS -Wno-unknown-pragmas"])

# Checks for things that fail under Emscripten, but won't prevent compilation
if test x$emscripten = xfalse; then  
  AC_SEARCH_LIBS([dlopen], [dl], [], [])
  AC_CHECK_LIB([atomic], [main], [], [], [])
  AC_CHECK_LIB([omp], [main], [], [], [])
  AC_CHECK_LIB([boost_filesystem], [main], [], [AC_CHECK_LIB([boost_filesystem-mt], [main], [], [AC_MSG_ERROR([Boost.FileSystem library not found])])])
  AC_CHECK_LIB([boost_system], [main], [], [AC_CHECK_LIB([boost_system-mt], [main], [], [AC_MSG_ERROR([Boost.System library not found])])])

  # Checks for headers
  if test xPACKAGE_TARNAME != xbirch_standard; then
    AC_CHECK_HEADERS([bi/birch_standard.hpp], [], [AC_MSG_ERROR([Birch standard library header not found.])], [-])
  fi
  AC_CHECK_HEADERS([omp.h], [], [], [-])
fi

# Checks for libraries
if test xPACKAGE_TARNAME != xbirch_standard; then
  AC_CHECK_LIB([birch_standard], [main], [], [AC_MSG_ERROR([Birch standard library not found])])
fi

# Checks for functions
AX_GCC_BUILTIN([__builtin_clz])
AX_GCC_BUILTIN([__builtin_clzll])
