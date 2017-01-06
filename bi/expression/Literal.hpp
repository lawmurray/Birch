/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/VarParameter.hpp"

#include <string>

namespace bi {
/**
 * Literal.
 *
 * @ingroup compiler_expression
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
      shared_ptr<Location> loc = nullptr);

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

  virtual bool dispatch(Expression& o);
  virtual bool le(Literal<T1>& o);
  virtual bool le(VarParameter& o);
};

/**
 * Boolean literal.
 *
 * @ingroup compiler_expression
 */
typedef Literal<bool> BoolLiteral;

/**
 * Integer literal.
 *
 * @ingroup compiler_expression
 */
typedef Literal<int64_t> IntLiteral;

/**
 * Floating point literal.
 *
 * @ingroup compiler_expression
 */
typedef Literal<double> RealLiteral;

/**
 * String literal.
 *
 * @ingroup compiler_expression
 */
typedef Literal<std::string> StringLiteral;
}
