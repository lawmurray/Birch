/**
 * @file
 */
#pragma once

#include "bi/common/Typed.hpp"
#include "bi/common/Located.hpp"

namespace bi {
class Cloner;
class Modifier;
class Visitor;

/**
 * Expression.
 *
 * @ingroup compiler_expression
 */
class Expression: public Located, public Typed {
public:
  /**
   * Constructor.
   *
   * @param type Type.
   * @param loc Location.
   */
  Expression(Type* type, shared_ptr<Location> loc = nullptr);

  /**
   * Constructor.
   *
   * @param loc Location.
   */
  Expression(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Expression() = 0;

  /**
   * Accept cloning visitor.
   *
   * @param v The visitor.
   *
   * @return Cloned (and potentially modified) expression.
   */
  virtual Expression* acceptClone(Cloner* visitor) const = 0;

  /**
   * Accept modifying visitor.
   *
   * @param v The visitor.
   *
   * @return Modified expression.
   */
  virtual void acceptModify(Modifier* visitor) = 0;

  /**
   * Accept read-only visitor.
   *
   * @param v The visitor.
   */
  virtual void accept(Visitor* visitor) const = 0;

  /*
   * Bool cast to check for non-empty expression.
   */
  virtual operator bool() const;

  /**
   * Is this a primary expression?
   */
  virtual bool isPrimary() const;

  /**
   * Is this a rich expression?
   */
  virtual bool isRich() const;

  /**
   * Strip parentheses.
   */
  virtual Expression* strip();

  /**
   * Number of expressions in tuple.
   */
  virtual int tupleSize() const;

  /**
   * Number of range expressions in tuple.
   */
  virtual int tupleDims() const;

  /*
   * Partial order comparison operators for comparing expressions in terms of
   * specialisation.
   *
   * The first two are the most commonly used, and so overridden by derived
   * classes. The remainder are expressed in terms of these.
   */
  virtual bool operator<=(Expression& o) = 0;
  virtual bool operator==(const Expression& o) const = 0;
  bool operator<(Expression& o);
  bool operator>(Expression& o);
  bool operator>=(Expression& o);
  bool operator!=(Expression& o);
};
}

inline bi::Expression::~Expression() {
  //
}

inline bi::Expression::operator bool() const {
  return true;
}

inline bool bi::Expression::operator<(Expression& o) {
  return *this <= o && o != *this;
}

inline bool bi::Expression::operator>(Expression& o) {
  return o <= *this && o != *this;
}

inline bool bi::Expression::operator>=(Expression& o) {
  return o <= *this;
}

inline bool bi::Expression::operator!=(Expression& o) {
  return !(*this == o);
}
