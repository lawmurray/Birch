/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<arithmetic To, argument Middle>
struct Cast {
  BIRCH_UNARY_FORM(Cast)
  BIRCH_UNARY_SIZE(Cast)
  BIRCH_UNARY_EVAL(Cast, cast<To>)
  BIRCH_UNARY_GRAD(Cast, cast_grad<To>)
};

template<arithmetic To, argument Middle>
struct is_form<Cast<To,Middle>> {
  static constexpr bool value = true;
};

template<arithmetic To, argument Middle>
struct tag_s<Cast<To,Middle>> {
  using type = Cast<To,tag_t<Middle>>;
};

template<arithmetic To, argument Middle>
struct peg_s<Cast<To,Middle>> {
  using type = Cast<To,peg_t<Middle>>;
};

template<class To, argument Middle>
auto cast(Middle&& m) {
  using TagMiddle = tag_t<Middle>;
  return Cast<To,TagMiddle>(std::in_place, std::forward<Middle>(m));
}

}
