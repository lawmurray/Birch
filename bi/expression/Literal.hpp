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

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(Literal<T1>& o);
  virtual bool definitely(VarParameter& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(Literal<T1>& o);
  virtual bool possibly(VarParameter& o);
};

/**
 * Boolean literal.
 *
 * @ingroup compiler_expression
 */
typedef Literal<bool> BooleanLiteral;

/**
 * Integer literal.
 *
 * @ingroup compiler_expression
 */
typedef Literal<int64_t> IntegerLiteral;

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
typedef Literal<const char*> StringLiteral;
}
