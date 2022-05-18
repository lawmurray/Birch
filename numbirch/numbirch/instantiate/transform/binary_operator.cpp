/**
 * @file
 * 
 * As binary.cpp, but removes the overload with only arithmetic types, as this
 * is disallowed by C++ (built-in operators are always used in this situation,
 * gcc error: "must have an argument of class or enumerated type").
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#include "numbirch/cuda/random.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#include "numbirch/eigen/random.hpp"
#endif
#include "numbirch/common/transform.hpp"

#define BINARY(f, R) \
    BINARY_FIRST(f, R, real) \
    BINARY_FIRST(f, R, int) \
    BINARY_FIRST(f, R, bool)
#define BINARY_FIRST(f, R, T) \
    BINARY_SECOND(f, R, T, real) \
    BINARY_SECOND(f, R, T, int) \
    BINARY_SECOND(f, R, T, bool)
#define BINARY_SECOND(f, R, T, U) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), U) \
    BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 0)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), U) \
    BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 2)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), U) \
    BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 1))
#define BINARY_SIG(f, R, T, U) \
    template R<T,U> f<T,U,int>(const T&, const U&);

#define BINARY_ARITHMETIC(f) BINARY(f, implicit_t)
#define BINARY_REAL(f) BINARY(f, real_t)
#define BINARY_BOOL(f) BINARY(f, bool_t)

namespace numbirch {
BINARY_BOOL(operator&&)
BINARY_BOOL(operator||)
BINARY_BOOL(operator==)
BINARY_BOOL(operator!=)
BINARY_BOOL(operator<)
BINARY_BOOL(operator<=)
BINARY_BOOL(operator>)
BINARY_BOOL(operator>=)
}
