AC_INIT([Birch], [0.0.0], [], [birch], [])
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

AC_ARG_ENABLE([assert],
     [  --enable-assert         enable assertion checking],
     [case "${enableval}" in
       yes) assert=true ;;
       no)  assert=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-assert]) ;;
     esac],[assert=true])

AC_ARG_ENABLE([extradebug],
     [  --enable-extradebug    use extra debugging options],
     [case "${enableval}" in
       yes) extradebug=true ;;
       no)  extradebug=false ;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-extradebug]) ;;
     esac],[extradebug=false])

AC_PROG_CXXCPP
AC_PROG_CXX
AC_PATH_PROG([BIRCHC], [birchc], [./birchc])  # assumes will be compiled into working directory if not found
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AX_CXX_COMPILE_STDCXX_14([noext], [mandatory])

# Checks for libraries
AC_CHECK_LIB([birch], [bi_present_], [], [AC_MSG_ERROR([Birch compiler library not found])])
if test x$std = xtrue; then
    AC_CHECK_LIB([birch_standard], [bi_standard_present_], [], [AC_MSG_ERROR([Birch standard library not found])])
fi

# Checks for header files
AC_CHECK_HEADERS([bi/bi.hpp], [], [AC_MSG_ERROR([Birch compiler library header not found.])], [-])
if test x$std = xtrue; then
    AC_CHECK_HEADERS([bi/standard.hpp], [], [AC_MSG_ERROR([Birch standard library header not found.])], [-])
fi

# Defines
AM_CONDITIONAL([ENABLE_STD], [test x$std = xtrue])
AM_CONDITIONAL([ENABLE_ASSERT], [test x$assert = xtrue])
AM_CONDITIONAL([ENABLE_EXTRADEBUG], [test x$extradebug = xtrue])

# Checks for typedefs, structures, and compiler characteristics
AC_HEADER_STDBOOL
AC_C_INLINE

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
