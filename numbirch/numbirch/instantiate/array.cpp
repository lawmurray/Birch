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
#include "numbirch/common/array.inl"
#include "numbirch/instantiate/instantiate.hpp"

namespace numbirch {

[[maybe_unused]] static void instantiate() {
  std::visit([]<class T>(T x) {
    diagonal(x, 0);
    fill(x, 0, 0);
    fill(x, 0);
    iota(x, 0);

    Array<real,2> gm;
    Array<real,1> gv;
    diagonal_grad(gm, x, 0);
    fill_grad(gm, x, 0, 0);
    fill_grad(gv, x, 0);
    iota_grad(gv, x, 0);
  }, scalar_variant());

  std::visit([]<class T>(T x) {
    diagonal(x);

    Array<real,2> g;
    diagonal_grad(g, x);
  }, vector_variant());

  std::visit([]<class T>(T x) {
    scal(x);
    vec(x);
    mat(x, 0);

    Array<real,0> gs;
    Array<real,1> gv;
    Array<real,2> gm;
    scal_grad(gs, x);
    vec_grad(gv, x);
    mat_grad(gm, x, 0);
  }, numeric_variant());

  std::visit([]<class T, class U, class V>(T x, U y, V z) {
    element(x, y, z);

    Array<real,0> g;
    element_grad1(g, x, y, z);
    element_grad2(g, x, y, z);
    element_grad3(g, x, y, z);
  }, matrix_variant(), scalar_variant(), scalar_variant());

  std::visit([]<class T, class U>(T x, U y) {
    element(x, y);

    Array<real,0> g;
    element_grad1(g, x, y);
    element_grad2(g, x, y);
  }, vector_variant(), scalar_variant());

  std::visit([]<class T, class U, class V>(T x, U y, V z) {
    single(x, y, z, 0, 0);

    Array<real,2> g;
    single_grad1(g, x, y, z, 0, 0);
    single_grad2(g, x, y, z, 0, 0);
    single_grad3(g, x, y, z, 0, 0);
  }, scalar_variant(), scalar_variant(), scalar_variant());

  std::visit([]<class T, class U>(T x, U y) {
    single(x, y, 0);

    Array<real,1> g;
    single_grad1(g, x, y, 0);
    single_grad2(g, x, y, 0);
  }, scalar_variant(), scalar_variant());

  std::visit([]<class T, class U>(T x, U y) {
    pack(x, y);
    stack(x, y);

    pack_grad1(real_t<pack_t<T,U>>(), x, y);
    pack_grad2(real_t<pack_t<T,U>>(), x, y);
    stack_grad1(real_t<stack_t<T,U>>(), x, y);
    stack_grad2(real_t<stack_t<T,U>>(), x, y);
  }, numeric_variant(), numeric_variant());

  std::visit([]<class T, class U, class V>(T x, U y, V z) {
    gather(x, y, z);
    scatter(x, y, z, 0, 0);

    Array<real,2> g;
    gather_grad1(g, x, y, z);
    gather_grad2(g, x, y, z);
    gather_grad3(g, x, y, z);
    scatter_grad1(g, x, y, z, 0, 0);
    scatter_grad2(g, x, y, z, 0, 0);
    scatter_grad3(g, x, y, z, 0, 0);
  }, matrix_variant(), matrix_variant(), matrix_variant());

  std::visit([]<class T, class U>(T x, U y) {
    gather(x, y);
    scatter(x, y, 0);

    Array<real,1> g;
    gather_grad1(g, x, y);
    gather_grad2(g, x, y);
    scatter_grad1(g, x, y, 0);
    scatter_grad2(g, x, y, 0);
  }, vector_variant(), vector_variant());

}

}
