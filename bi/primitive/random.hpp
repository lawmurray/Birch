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
   * @param expr Model expression.
   * @param pull Pull expression.
   * @param push Push expression.
   */
  random(Variate& x, std::function<Model()> expr, std::function<void()> pull,
      std::function<void()> push);

  /**
   * Destructor.
   */
  ~random();

  /**
   * Cast to variate.
   */
  operator Variate&();

private:
  Variate& x;
  std::function<Model()> expr;
  std::function<void()> pull;
  std::function<void()> push;

  bool pulled;
  bool pushed;
};
}

template<class Variate, class Model>
bi::random<Variate,Model>::random(Variate& x, std::function<Model()> expr,
    std::function<void()> pull, std::function<void()> push) :
    x(x),
    expr(expr),
    pull(pull),
    push(push),
    pulled(false),
    pushed(false) {
  //
}

template<class Variate, class Model>
bi::random<Variate,Model>::~random() {
  if (!pulled) {
    pull();
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
    pull();
    pulled = true;
  }
  return x;
}
