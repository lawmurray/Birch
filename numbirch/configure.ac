AC_INIT([NumBirch], [unversioned], [], [numbirch], [https://birch.sh])
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

AC_ARG_ENABLE([default],
[AS_HELP_STRING[--enable-default], [Use default]],
[case "${enableval}" in
  yes) default=true ;;
  no)  default=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-default]) ;;
esac],[default=true])

# Programs
if $oneapi; then
  CC=./dpcpp_wrapper
  CXX=./dpcpp_wrapper
else
  if $cuda; then
    CC=./nvcc_wrapper
    CXX=./nvcc_wrapper
  fi
fi
AC_PROG_CXX
AC_PROG_CXXCPP
AC_PROG_INSTALL

AC_DISABLE_STATIC
AC_ENABLE_SHARED
LT_INIT([dlopen,pic-only])

# Checks for basic things
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
AC_HEADER_ASSERT

# OpenMP checks, -fiopenmp is for the Intel compiler and library, check first
AX_CHECK_COMPILE_FLAG([-fiopenmp], [CXXFLAGS="$CXXFLAGS -fiopenmp"], [], [-Werror])
AC_OPENMP
AC_CHECK_HEADERS([omp.h], [], [], [AC_INCLUDES_DEFAULT])
AC_SEARCH_LIBS([omp_get_max_threads], [omp])

# Checks for compiler flags
AX_CHECK_COMPILE_FLAG([-fprofile-abs-path], [CXXFLAGS="$CXXFLAGS -fprofile-abs-path"], [], [-Werror])

# Check for build specifics
if $oneapi; then
  echo
else
  if $cuda; then
    AC_CHECK_HEADERS([cublas_v2.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([cusolverDn.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([thrust/execution_policy.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([thrust/copy.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([thrust/transform.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([thrust/transform_reduce.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([thrust/inner_product.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([thrust/iterator/counting_iterator.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([thrust/iterator/transform_iterator.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
    AC_CHECK_HEADERS([jemalloc/jemalloc.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])

    AC_CHECK_LIB([cublas], [main], [], [AC_MSG_ERROR([required library not found.])], [])
    AC_CHECK_LIB([cublasLt], [main], [], [AC_MSG_ERROR([required library not found.])], [])
    AC_CHECK_LIB([cusolver], [main], [], [AC_MSG_ERROR([required library not found.])], [])
    AC_CHECK_LIB([jemalloc], [malloc], [], [AC_MSG_ERROR([required library not found.])], [])
  else
    AC_CHECK_HEADERS([eigen3/Eigen/Dense], [], [AC_MSG_ERROR([required Eigen header not found.])], [AC_INCLUDES_DEFAULT])
  fi
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
