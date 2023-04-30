AC_INIT([NumBirch], [0.0.0], [], [numbirch], [https://birch.sh])
AC_PREREQ([2.62])
AC_LANG([C++])
AM_INIT_AUTOMAKE([1.10 no-define foreign subdir-objects])
AC_CONFIG_MACRO_DIR([m4])

AC_ARG_ENABLE([checks],
[AS_HELP_STRING[--enable-checks], [Enable header and library checks, can disable if just running make dist]],
[case "${enableval}" in
  yes) checks=true ;;
  no)  checks=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-checks]) ;;
esac],[checks=true])

AC_ARG_ENABLE([oneapi],
[AS_HELP_STRING[--enable-oneapi], [Use DPC++]],
[case "${enableval}" in
  yes) oneapi=true ;;
  no)  oneapi=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-oneapi]) ;;
esac],[oneapi=false])
AM_CONDITIONAL([ONEAPI], [test x$oneapi = xtrue])

AC_ARG_ENABLE([cuda],
[AS_HELP_STRING[--enable-cuda], [Use CUDA]],
[case "${enableval}" in
  yes) cuda=true ;;
  no)  cuda=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-cuda]) ;;
esac],[cuda=false])
AM_CONDITIONAL([CUDA], [test x$cuda = xtrue])

AC_ARG_ENABLE([eigen],
[AS_HELP_STRING[--enable-eigen], [Use Eigen]],
[case "${enableval}" in
  yes) eigen=true ;;
  no)  eigen=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-eigen]) ;;
esac],[eigen=true])
AM_CONDITIONAL([EIGEN], [test x$eigen = xtrue])

AC_ARG_ENABLE([double],
[AS_HELP_STRING[--enable-double], [Build double precision libraries]],
[case "${enableval}" in
  yes) double=true ;;
  no)  double=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-double]) ;;
esac],[double=true])
AM_CONDITIONAL([DOUBLE], [test x$double = xtrue])

AC_ARG_ENABLE([single],
[AS_HELP_STRING[--enable-single], [Build single precision libraries]],
[case "${enableval}" in
  yes) single=true ;;
  no)  single=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-single]) ;;
esac],[single=true])
AM_CONDITIONAL([SINGLE], [test x$single = xtrue])

# Programs
if $oneapi; then
  CXX=dpcpp_wrapper
else
  if $cuda; then
    CXX=nvcc_wrapper
  fi
fi
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_INSTALL
LT_INIT([dlopen disable-static])

if $checks; then
  # Checks for basic things
  AX_CXX_COMPILE_STDCXX(20, [noext], [mandatory])
  AC_HEADER_ASSERT

  # Checks for common requirements
  AC_CHECK_LIB([atomic], [main], [], [], [])
  AC_OPENMP
  CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"
  AS_IF([test "$OPENMP_CXXFLAGS" != ""], [AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])])
  AS_IF([test "$OPENMP_CXXFLAGS" != ""], [AC_SEARCH_LIBS([omp_get_thread_num], [omp], [], [], [])])
  AC_CHECK_HEADERS([eigen3/unsupported/Eigen/SpecialFunctions], [], [AC_CHECK_HEADERS([unsupported/Eigen/SpecialFunctions], [], [AC_MSG_ERROR([required Eigen header not found.])], [AC_INCLUDES_DEFAULT])], [AC_INCLUDES_DEFAULT])

  # Checks for build-specific requirements
  if $oneapi; then
    AC_CHECK_LIB([jemalloc_numbirch_pic], [numbirch_malloc], [], [AC_MSG_ERROR([required library not found.])], [])
  fi
  if $cuda; then
    AC_CHECK_HEADERS([cublas_v2.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([cusolverDn.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([jemalloc/jemalloc_numbirch.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_LIB([cublas], [main], [], [AC_MSG_ERROR([required library not found.])], [])
    AC_CHECK_LIB([cusolver], [main], [], [AC_MSG_ERROR([required library not found.])], [])
    AC_CHECK_LIB([jemalloc_numbirch_pic], [numbirch_malloc], [], [AC_MSG_ERROR([required library not found.])], [])
  fi
  if $eigen; then
    AC_CHECK_HEADERS([eigen3/Eigen/Dense], [], [AC_CHECK_HEADERS([Eigen/Dense], [], [AC_MSG_ERROR([required Eigen header not found.])], [AC_INCLUDES_DEFAULT])], [AC_INCLUDES_DEFAULT])
  fi
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
