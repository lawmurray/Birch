/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
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
   * @param value Value.
   * @param str Preferred string encoding of @p value.
   * @param type Type.
   * @param loc Location.
   */
  Literal(const T1& value, const std::string& str, Type* type,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Literal();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Value.
   */
  T1 value;

  /**
   * Preferred string encoding of value.
   */
  std::string str;
};
}
