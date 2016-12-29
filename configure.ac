AC_INIT([Birch], [0.0.0], [], [birch], [])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign])
#AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

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
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_PROG_LEX
if test "x$LEX" = "x:"; then
  AC_MSG_ERROR([Flex not found.])
fi
AC_PROG_YACC
if test "x$YACC" = "xyacc"; then
  AC_MSG_ERROR([Bison not found.])
fi
AX_CXX_COMPILE_STDCXX_14([noext], [mandatory])

# Compiler characteristics                                                      
if test x$openmp = xtrue; then
  AC_OPENMP
fi

# Checks for libraries
AC_CHECK_LIB([m], [main], [], [AC_MSG_ERROR([required standard math library not found])])
AC_CHECK_LIB([netcdf], [main], [], [AC_MSG_ERROR([required NetCDF library not found])])
AC_CHECK_LIB([boost_filesystem], [main], [], [AC_CHECK_LIB([boost_filesystem-mt], [main], [], [AC_MSG_ERROR([Boost.FileSystem library not found])])])
AC_CHECK_LIB([boost_system], [main], [], [AC_CHECK_LIB([boost_system-mt], [main], [], [AC_MSG_ERROR([Boost.System library not found])])])

# Checks for header files
AC_HEADER_ASSERT

AC_CHECK_HEADERS([netcdf.h], [], \
    AC_MSG_ERROR([required NetCDF header not found]), [-])

AC_CHECK_HEADERS([\
    boost/filesystem.hpp \
    boost/filesystem/fstream.hpp \
    boost/interprocess/sync/file_lock.hpp \
    boost/algorithm/string.hpp \
    ], [], AC_MSG_ERROR([required Boost header not found]), [-])

# Defines
AM_CONDITIONAL([ENABLE_STD], [test x$std = xtrue])
AM_CONDITIONAL([ENABLE_ASSERT], [test x$assert = xtrue])
AM_CONDITIONAL([ENABLE_EXTRADEBUG], [test x$extradebug = xtrue])

# Checks for typedefs, structures, and compiler characteristics
AC_HEADER_STDBOOL
AC_C_INLINE

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
