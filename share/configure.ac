AC_INIT([PACKAGE_NAME], [0.0.0], [], [PACKAGE_TARNAME], [])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
LT_INIT([dlopen])
#AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Conditionals used by make
AC_ARG_ENABLE([std],
     [  --enable-std            enable standard library],
     [case "${enableval}" in
       yes) std=true ;;
       no)  std=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-std]) ;;
     esac],[std=true])

AC_PROG_CXXCPP
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AX_CXX_COMPILE_STDCXX_14([noext], [mandatory])

# Checks for libraries
AC_CHECK_LIB([birch], [bi_present], [], [AC_MSG_ERROR([Birch compiler library not found])])
AC_CHECK_LIB([profiler], [main], [], [])
if test x$std = xtrue; then
    AC_CHECK_LIB([birch_standard], [bi_standard_present], [], [AC_MSG_ERROR([Birch standard library not found])])
fi

# Checks for header files
AC_CHECK_HEADERS([bi/libbirch.hpp], [], [AC_MSG_ERROR([Birch compiler library header not found.])], [-])
if test x$std = xtrue; then
    AC_CHECK_HEADERS([bi/standard.hpp], [], [AC_MSG_ERROR([Birch standard library header not found.])], [-])
fi

# Defines
AM_CONDITIONAL([ENABLE_STD], [test x$std = xtrue])

# Checks for typedefs, structures, and compiler characteristics
AC_HEADER_STDBOOL
AC_C_INLINE

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
