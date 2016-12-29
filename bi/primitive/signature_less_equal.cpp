/**
 * @file
 */
#include "bi/primitive/signature_less_equal.hpp"

#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/FuncReference.hpp"

bool bi::signature_less_equal::operator()(FuncParameter* o1,
    FuncParameter* o2) {
  return *o1->parens <= *o2->parens;
}

bool bi::signature_less_equal::operator()(const FuncReference* o1,
    FuncParameter* o2) {
  return *o1->parens <= *o2->parens;
}
