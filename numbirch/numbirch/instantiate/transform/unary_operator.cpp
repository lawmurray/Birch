/**
 * @file
 * 
 * As unary.cpp, but removes the overload with only arithmetic types, as this
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

#define UNARY(f, R) \
    UNARY_FIRST(f, R, real) \
    UNARY_FIRST(f, R, int) \
    UNARY_FIRST(f, R, bool)
#define UNARY_FIRST(f, R, T) \
    UNARY_SIG(f, R, ARRAY(T, 2)) \
    UNARY_SIG(f, R, ARRAY(T, 1)) \
    UNARY_SIG(f, R, ARRAY(T, 0))
#define UNARY_SIG(f, R, T) \
    template R<T> f<T,int>(const T&);

#define UNARY_ARITHMETIC(f) UNARY(f, implicit_t)
#define UNARY_REAL(f) UNARY(f, real_t)
#define UNARY_BOOL(f) UNARY(f, bool_t)

namespace numbirch {
UNARY_BOOL(operator!)
}
