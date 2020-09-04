/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"

namespace birch {
/**
 * Literal.
 *
 * @ingroup expression
 */
template<class T1>
class Literal: public Expression {
public:
  /**
   * Constructor.
   *
   * @param str String encoding of @p value.
   * @param loc Location.
   */
  Literal(const std::string& str, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Literal();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Preferred string encoding of value.
   */
  std::string str;
};
}
