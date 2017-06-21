/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/Function.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Placeholder for identifier to unknown object.
 */
struct Unknown {
  //
};

/**
 * Identifier.
 *
 * @ingroup compiler_expression
 *
 * @tparam ObjectType The particular type of object referred to by the
 * identifier.
 *
 * An identifier refers to a variable, function or the like. For most
 * identifiers, the type of object referred to is ambiguous in syntax:
 * for example, @c f() may call a function or a variable of lambda type. When
 * parsing, such identifiers are given the type @c Identifier<Unknown>, and
 * are later replaced, by Resolver, with an identifier of the appropriate
 * type.
 */
template<class ObjectType = Unknown>
class Identifier: public Expression,
    public Named,
    public Parenthesised,
    public Reference<ObjectType> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Expression in parentheses.
   * @param loc Location.
   * @param target Target.
   */
  Identifier(shared_ptr<Name> name, Expression* parens =
      new EmptyExpression(), shared_ptr<Location> loc = nullptr,
      const ObjectType* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~Identifier();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const Identifier<ObjectType>& o) const;
  virtual bool definitely(const Parameter& o) const;
  virtual bool definitely(const GlobalVariable& o) const;
  virtual bool definitely(const LocalVariable& o) const;
  virtual bool definitely(const MemberVariable& o) const;
  virtual bool definitely(const Function& o) const;
  virtual bool definitely(const Coroutine& o) const;
  virtual bool definitely(const MemberFunction& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const Identifier<ObjectType>& o) const;
  virtual bool possibly(const Parameter& o) const;
  virtual bool possibly(const GlobalVariable& o) const;
  virtual bool possibly(const LocalVariable& o) const;
  virtual bool possibly(const MemberVariable& o) const;
  virtual bool possibly(const Function& o) const;
  virtual bool possibly(const Coroutine& o) const;
  virtual bool possibly(const MemberFunction& o) const;
};

/**
 * Specialisation for usage of binary operators. Such usage is clear by
 * syntax.
 *
 * @ingroup compiler_expression
 */
template<>
class Identifier<BinaryOperator> : public Expression,
    public Named,
    public Binary<Expression>,
    public Reference<BinaryOperator> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param name Name.
   * @param right Right operand.
   * @param loc Location.
   * @param target Target.
   */
  Identifier(Expression* left, shared_ptr<Name> name, Expression* right,
      shared_ptr<Location> loc = nullptr, const BinaryOperator* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~Identifier();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const Identifier<BinaryOperator>& o) const;
  virtual bool definitely(const BinaryOperator& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const Identifier<BinaryOperator>& o) const;
  virtual bool possibly(const BinaryOperator& o) const;
  virtual bool possibly(const Parameter& o) const;
};

/**
 * Specialisation for usage of unary operators. Such usage is clear by
 * syntax.
 *
 * @ingroup compiler_expression
 */
template<>
class Identifier<UnaryOperator> : public Expression,
    public Named,
    public Unary<Expression>,
    public Reference<UnaryOperator> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param single Operand.
   * @param loc Location.
   * @param target Target.
   */
  Identifier(shared_ptr<Name> name, Expression* single,
      shared_ptr<Location> loc = nullptr, const UnaryOperator* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~Identifier();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const Identifier<UnaryOperator>& o) const;
  virtual bool definitely(const UnaryOperator& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const Identifier<UnaryOperator>& o) const;
  virtual bool possibly(const UnaryOperator& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
