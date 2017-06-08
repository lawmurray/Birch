/**
 * @file
 */
#include "bi/common/Signature.hpp"

#include "bi/common/List.hpp"
#include "bi/primitive/encode.hpp"

bi::Signature::Signature(shared_ptr<Name> name, Expression* parens,
    const FunctionForm form) :
    Named(name),
    Parenthesised(parens),
    FunctionMode(form) {
  //
}

bi::Signature::~Signature() {
  //
}
