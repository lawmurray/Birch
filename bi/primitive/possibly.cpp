/**
 * @file
 */
#include "bi/primitive/possibly.hpp"

#include <algorithm>

bi::possibly bi::possibly::operator&&(const possibly& o) const {
  return possibly(std::min(state, o.state));
}

bi::possibly bi::possibly::operator||(const possibly& o) const {
  return possibly(std::max(state, o.state));
}

bi::possibly bi::possibly::operator&&(const bool& o) const {
  return *this && possibly(o);
}

bi::possibly bi::possibly::operator||(const bool& o) const {
  return *this || possibly(o);
}

bi::possibly bi::possibly::operator!() const {
  return possibly(DEFINITE - state);
}

//bi::possibly bi::operator&&(const bool& o1, const possibly& o2) {
//  return possibly(o1) && o2;
//}

//bi::possibly bi::operator||(const bool& o1, const possibly& o2) {
//  return possibly(o1) || o2;
//}
