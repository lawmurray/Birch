/**
 * @file
 */
#pragma once

#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/FunctionMode.hpp"

namespace bi {
/**
 * Signature of a function.
 *
 * @ingroup compiler_expression
 */
class Signature: public Named,
    public Numbered,
    public Parenthesised,
    public FunctionMode {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses expression.
   * @param form Function form.
   */
  Signature(shared_ptr<Name> name, Expression* parens,
      const FunctionForm form = FUNCTION_FORM);

  /**
   * Destructor.
   */
  virtual ~Signature() = 0;
};
}
