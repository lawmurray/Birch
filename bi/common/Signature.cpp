/**
 * @file
 */
#include "bi/common/Signature.hpp"

#include "bi/common/List.hpp"
#include "bi/primitive/encode.hpp"

bi::Signature::Signature(shared_ptr<Name> name, Expression* parens,
    Expression* result, const SignatureForm form) :
    Named(name),
    Parenthesised(parens),
    Formed(form),
    result(result) {
  //
}

bi::Signature::~Signature() {
  //
}
