/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

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
  Literal(const T1& value, const std::string& str, Type* type, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Literal();

  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;

  /**
   * Value.
   */
  T1 value;

  /**
   * Preferred string encoding of value.
   */
  std::string str;
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
typedef Literal<int32_t> IntLiteral;

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

template<class T1>
inline bi::Literal<T1>::~Literal() {
  //
}
