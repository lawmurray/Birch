/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/transform.inl"

#define WHERE(f) \
    WHERE_FIRST(f, real) \
    WHERE_FIRST(f, int) \
    WHERE_FIRST(f, bool)
#define WHERE_FIRST(f, T) \
    WHERE_SECOND(f, T, real) \
    WHERE_SECOND(f, T, int) \
    WHERE_SECOND(f, T, bool)
#define WHERE_SECOND(f, T, U) \
    WHERE_THIRD(f, T, U, real) \
    WHERE_THIRD(f, T, U, int) \
    WHERE_THIRD(f, T, U, bool)
#define WHERE_THIRD(f, T, U, V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2), NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1), NUMBIRCH_ARRAY(V, 1)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0), V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), U, NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), U, V) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 0), V) \
    WHERE_SIG(f, T, U, NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, T, U, V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2), V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), U, NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0), V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), U, NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 2), U, V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2), NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 2), NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2), V) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 2), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 2), V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), U, NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, T, U, NUMBIRCH_ARRAY(V, 2)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1), V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 1)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), U, NUMBIRCH_ARRAY(V, 1)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0), V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), U, NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 1), U, V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1), NUMBIRCH_ARRAY(V, 1)) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 1), NUMBIRCH_ARRAY(V, 1)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1), V) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 1), NUMBIRCH_ARRAY(V, 0)) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 1), V) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 1)) \
    WHERE_SIG(f, NUMBIRCH_ARRAY(T, 0), U, NUMBIRCH_ARRAY(V, 1)) \
    WHERE_SIG(f, T, NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 1)) \
    WHERE_SIG(f, T, U, NUMBIRCH_ARRAY(V, 1))
#define WHERE_SIG(f, T, U, V) \
    template implicit_t<T,U,V> f<T,U,V>(const T&, const U&, const V&); \
    template real_t<T> f ## _grad1<T,U,V>(const real_t<U,V>& g, \
        const implicit_t<T,U,V>& r, const T& x, const U& y, const V& z); \
    template real_t<U> f ## _grad2<T,U,V>(const real_t<U,V>& g, \
        const implicit_t<T,U,V>& r, const T& x, const U& y, const V& z); \
    template real_t<V> f ## _grad3<T,U,V>(const real_t<U,V>& g, \
        const implicit_t<T,U,V>& r, const T& x, const U& y, const V& z); \

namespace numbirch {
WHERE(where)
}
