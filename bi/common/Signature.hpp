/**
 * @file
 */
#pragma once

#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Mangled.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Formed.hpp"

namespace bi {
/**
 * Signature of a function, operator, dispatcher, etc.
 *
 * @ingroup compiler_expression
 */
class Signature: public Named,
    public Numbered,
    public Mangled,
    public Parenthesised,
    public Formed {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses expression.
   * @param result Result expression.
   * @param form Function form.
   */
  Signature(shared_ptr<Name> name, Expression* parens, Expression* result,
      const SignatureForm form = FUNCTION);

  /**
   * Destructor.
   */
  virtual ~Signature() = 0;

  /**
   * Result expression.
   */
  unique_ptr<Expression> result;
};
}
