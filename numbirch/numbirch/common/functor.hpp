/**
 * @file
 * 
 * Functors used by multiple backends.
 */
#pragma once

#ifdef BACKEND_CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace numbirch {
template<class T>
struct vector_element_functor {
  vector_element_functor(T* x, int incx) :
      x(x),
      incx(incx) {
    //
  }
  HOST_DEVICE T operator()(const int i) const {
    return x[i*incx];
  }
  T* x;
  int incx;
};

template<class T>
struct matrix_element_functor {
  matrix_element_functor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  HOST_DEVICE T operator()(const int i) const {
    int c = i/m;
    int r = i - c*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T>
struct matrix_transpose_element_functor {
  matrix_transpose_element_functor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  HOST_DEVICE T operator()(const int i) const {
    int r = i/m;
    int c = i - r*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T = double>
struct negate_functor {
  HOST_DEVICE T operator()(const T x) const {
    return -x;
  }
};

template<class T = double>
struct rectify_functor {
  HOST_DEVICE T operator()(const T x) const {
    return x > T(0) ? x : T(0);
  }
};

template<class T = double>
struct plus_functor {
  HOST_DEVICE T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<class T = double>
struct minus_functor {
  HOST_DEVICE T operator()(const T x, const T y) const {
    return x - y;
  }
};

template<class T = double>
struct multiplies_functor {
  HOST_DEVICE T operator()(const T x, const T y) const {
    return x*y;
  }
};

template<class T = double>
struct divides_functor {
  HOST_DEVICE T operator()(const T x, const T y) const {
    return x/y;
  }
};

template<class T = double>
struct scalar_multiplies_functor {
  scalar_multiplies_functor(T a) :
      a(a) {
    //
  }
  HOST_DEVICE T operator()(const T x) const {
    return x*a;
  }
  T a;
};

template<class T = double>
struct scalar_divides_functor {
  scalar_divides_functor(T a) :
      a(a) {
    //
  }
  HOST_DEVICE T operator()(const T x) const {
    return x/a;
  }
  T a;
};

template<class T = double>
struct combine_functor {
  combine_functor(const T a , const T b, const T c, const T d) :
      a(a), b(b), c(c), d(d) {
    //
  }
  HOST_DEVICE T operator()(const T w, const T x, const T y, const T z) const {
    return a*w + b*x + c*y + d*z;
  }
  const T a, b, c, d;
};

template<class T = double>
struct combine4_functor {
  combine4_functor(const T a , const T b, const T c, const T d) :
      a(a), b(b), c(c), d(d) {
    //
  }
  HOST_DEVICE T operator()(const std::tuple<T,T,T,T>& o) const {
    return a*std::get<0>(o) + b*std::get<1>(o) + c*std::get<2>(o) +
        d*std::get<3>(o);
  }
  const T a, b, c, d;
};

template<class T = double>
struct log_abs_functor {
  HOST_DEVICE T operator()(const T x) const {
    return std::log(std::abs(x));
  }
};

template<class T = double>
struct log_functor {
  HOST_DEVICE T operator()(const T x) const {
    return std::log(x);
  }
};

template<class T = double>
struct diagonal_functor {
  diagonal_functor(T a) :
      a(a) {
    //
  }
  HOST_DEVICE T operator()(const int i, const int j) const {
    return (i == j) ? a : T(0);
  }
  T a;
};

}