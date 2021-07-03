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

AC_ARG_ENABLE([dpcpp],
[AS_HELP_STRING[--enable-dpcpp], [Use DPC++]],
[case "${enableval}" in
  yes) dpcpp=true ;;
  no)  dpcpp=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-dpcpp]) ;;
esac],[dpcpp=false])
AM_CONDITIONAL([DPCPP], [test x$dpcpp = xtrue])

AC_ARG_ENABLE([cblas],
[AS_HELP_STRING[--enable-cblas], [Use CBLAS and LAPACKE (the C interfaces to BLAS and LAPACK)]],
[case "${enableval}" in
  yes) cblas=true ;;
  no)  cblas=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-cblas]) ;;
esac],[cblas=true])
AM_CONDITIONAL([CBLAS], [test x$cblas = xtrue])

AC_ARG_ENABLE([eigen],
[AS_HELP_STRING[--enable-eigen], [Use Eigen]],
[case "${enableval}" in
  yes) eigen=true ;;
  no)  eigen=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-eigen]) ;;
esac],[eigen=false])
AM_CONDITIONAL([EIGEN], [test x$eigen = xtrue])

# Programs
if $dpcpp; then
  compilers=dpcpp
else
  compilers=[]
fi
AC_PROG_CXX($compilers)
AC_PROG_CXXCPP
AC_PROG_INSTALL

AC_DISABLE_STATIC
AC_ENABLE_SHARED
AX_CXX_COMPILE_STDCXX(17, [noext], [mandatory])
LT_INIT([dlopen,pic-only])

# Checks for basic things
AC_HEADER_ASSERT
AC_HEADER_STDBOOL
AC_C_INLINE

# Checks for compiler flags
AX_CHECK_COMPILE_FLAG([-fprofile-abs-path], [CXXFLAGS="$CXXFLAGS -fprofile-abs-path"], [], [-Werror])

# Check for build specifics
if $dpcpp; then
  AX_CHECK_COMPILE_FLAG([-qmkl=sequential], [CXXFLAGS="$CXXFLAGS -qmkl=sequential"], [AC_MSG_ERROR([MKL not available])], [-Werror])
  AX_CHECK_COMPILE_FLAG([-fsycl], [CXXFLAGS="$CXXFLAGS -fsycl"], [AC_MSG_ERROR([SYCL not available])], [-Werror])
fi
if $cblas; then
  AC_CHECK_HEADERS([cblas.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
  AC_CHECK_HEADERS([lapacke.h], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
  AC_SEARCH_LIBS([cblas_dgemm], [openblas cblas], [], [AC_MSG_ERROR([required CBLAS function not found])], [])
  AC_SEARCH_LIBS([LAPACKE_dpotrf], [openblas lapacke], [], [AC_MSG_ERROR([required LAPACKE function not found])], [])
fi
if $eigen; then
  AC_CHECK_HEADERS([eigen3/Eigen/Dense], [], [AC_MSG_ERROR([required header not found.])], [AC_INCLUDES_DEFAULT])
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
