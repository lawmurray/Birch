/**
 * @file
 */
#include "bi/primitive/signature_less_equal.hpp"

#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/FuncReference.hpp"

bi::possibly bi::signature_less_equal::operator()(FuncParameter* o1,
    FuncParameter* o2) {
  return *o1 <= *o2;
}

bi::possibly bi::signature_less_equal::operator()(FuncReference* o1,
    FuncParameter* o2) {
  return *o1 <= *o2;
}
