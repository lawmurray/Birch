AC_INIT([PACKAGE_NAME], [PACKAGE_VERSION], [], [PACKAGE_TARNAME], [])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

# Default CXXFLAGS to empty, user can either override with their own, or use
# MODE to select a preset
: ${CXXFLAGS=""}
AC_ARG_VAR(MODE, [build mode for flag presets: debug, test, release, or empty.])
AM_CONDITIONAL([DEBUG], [test x$MODE = xdebug])
AM_CONDITIONAL([TEST], [test x$MODE = xtest])
AM_CONDITIONAL([RELEASE], [test x$MODE = xrelease])

# Suffix on installed libraries
AC_ARG_VAR(SUFFIX, [suffix for libaries, e.g. _debug, or empty.])

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
AC_OPENMP

# Checks for compiler flags
AX_CHECK_COMPILE_FLAG([-fno-caret-diagnostics], [CXXFLAGS="$CXXFLAGS -fno-caret-diagnostics"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-fno-diagnostics-show-option], [CXXFLAGS="$CXXFLAGS -fno-diagnostics-show-option"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-overloaded-virtual], [CXXFLAGS="$CXXFLAGS -Wno-overloaded-virtual"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-inconsistent-missing-override], [CXXFLAGS="$CXXFLAGS -Wno-inconsistent-missing-override"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-attributes], [CXXFLAGS="$CXXFLAGS -Wno-attributes"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-int-to-pointer-cast], [CXXFLAGS="$CXXFLAGS  -Wno-int-to-pointer-cast"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-unused-parameter], [CXXFLAGS="$CXXFLAGS -Wno-unused-parameter"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-unused-variable], [CXXFLAGS="$CXXFLAGS -Wno-unused-variable"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-unused-value], [CXXFLAGS="$CXXFLAGS -Wno-unused-value"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-unused-local-typedefs], [CXXFLAGS="$CXXFLAGS -Wno-unused-local-typedefs"], [], [-Werror])
AX_CHECK_COMPILE_FLAG([-Wno-unknown-pragmas], [CXXFLAGS="$CXXFLAGS -Wno-unknown-pragmas"], [], [-Werror])

# Checks for things that fail under Emscripten, but won't prevent compilation
if test x$emscripten = xfalse; then
  AC_SEARCH_LIBS([dlopen], [dl], [], [])
  AC_CHECK_LIB([atomic], [main], [], [], [])
  AC_CHECK_LIB([omp], [main], [], [], [])
  AC_CHECK_LIB([explain], [explain_system], [], [], [])
  AC_CHECK_LIB([boost_filesystem], [main], [], [AC_CHECK_LIB([boost_filesystem-mt], [main], [], [AC_MSG_ERROR([required library not found])])])
  AC_CHECK_LIB([boost_system], [main], [], [AC_CHECK_LIB([boost_system-mt], [main], [], [AC_MSG_ERROR([required library not found])])])
  AC_CHECK_LIB([birch$SUFFIX], [main], [], [AC_MSG_ERROR([required library not found])])

  # Checks for headers
  AC_CHECK_HEADERS([omp.h], [], [], [-])
  AC_CHECK_HEADERS([libexplain/system.h], [], [], [-])
  AC_CHECK_HEADERS([libbirch.hpp], [], [AC_MSG_ERROR([required header not found])], [-])
fi
