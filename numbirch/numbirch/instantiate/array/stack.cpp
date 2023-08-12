/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/array.inl"

#define STACK(f) \
    STACK_FIRST(f, real) \
    STACK_FIRST(f, int) \
    STACK_FIRST(f, bool)
#define STACK_FIRST(f, T) \
    STACK_SECOND(f, T, real) \
    STACK_SECOND(f, T, int) \
    STACK_SECOND(f, T, bool)
#define STACK_SECOND(f, T, U) \
    STACK_SIG(f, T, U) \
    STACK_SIG(f, T, NUMBIRCH_ARRAY(U, 0)) \
    STACK_SIG(f, T, NUMBIRCH_ARRAY(U, 1)) \
    STACK_SIG(f, T, NUMBIRCH_ARRAY(U, 2)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 0), U) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 1), U) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 2)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 2), U) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 1)) \
    STACK_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2))
#define STACK_SIG(f, T, U) \
    template stack_t<T,U> f(const T& x, const U& y); \
    template real_t<T> f##_grad1(const real_t<stack_t<T,U>>& g, \
        const T& x, const U& y); \
    template real_t<U> f##_grad2(const real_t<stack_t<T,U>>& g, \
        const T& x, const U& y);

namespace numbirch {
STACK(stack)
}
