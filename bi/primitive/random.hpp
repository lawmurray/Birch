/**
 * @file
 */
#pragma once

#include <functional>

namespace bi {
/**
 * Random variable.
 *
 * @tparam Variate Variate type.
 * @tparam Model Model type.
 */
template<class Variate, class Model>
class random {
public:
  /**
   * Constructor.
   *
   * @param x Variate.
   * @param m Model.
   * @param push Push expression.
   */
  random(Variate& x, const Model& m, std::function<void()> push);

  /**
   * Destructor.
   */
  ~random();

  /**
   * Cast to variate.
   */
  operator Variate&();

  Variate& x;
  Model m;

private:
  std::function<void()> push;

  bool pulled;
  bool pushed;
};
}

template<class Variate, class Model>
bi::random<Variate,Model>::random(Variate& x, const Model& m,
    std::function<void()> push) :
    x(x),
    m(m),
    push(push),
    pulled(false),
    pushed(false) {
  //
}

template<class Variate, class Model>
bi::random<Variate,Model>::~random() {
  if (!pulled) {
    pull_(x, m);
    pulled = true;
  }
  if (!pushed) {
    push();
    pushed = true;
  }
}

template<class Variate, class Model>
bi::random<Variate,Model>::operator Variate&() {
  if (!pulled) {
    pull_(x, m);
    pulled = true;
    push();
    pushed = true;
  }
  return x;
}
