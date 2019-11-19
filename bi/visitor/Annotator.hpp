/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/common/Annotated.hpp"

namespace bi {
/**
 * Visitor to add an annotation to the target of a reference.
 *
 * @ingroup visitor
 */
class Annotator : public Modifier {
public:
  /**
   * Constructor.
   *
   * @param a The annotation to add.
   */
  Annotator(const Annotation a);

  /**
   * Destructor.
   */
  virtual ~Annotator();

  Expression* modify(Identifier<Parameter>* o);
  Expression* modify(Identifier<GlobalVariable>* o);
  Expression* modify(Identifier<MemberVariable>* o);
  Expression* modify(Identifier<LocalVariable>* o);
  Expression* modify(Identifier<ForVariable>* o);
  Expression* modify(OverloadedIdentifier<Unknown>* o);
  Expression* modify(OverloadedIdentifier<Function>* o);
  Expression* modify(OverloadedIdentifier<Fiber>* o);
  Expression* modify(OverloadedIdentifier<MemberFiber>* o);
  Expression* modify(OverloadedIdentifier<MemberFunction>* o);
  Expression* modify(OverloadedIdentifier<BinaryOperator>* o);
  Expression* modify(OverloadedIdentifier<UnaryOperator>* o);

  Type* modify(ClassType* o);
  Type* modify(BasicType* o);
  Type* modify(GenericType* o);

private:
  /**
   * The annotation to add.
   */
  const Annotation a;
};
}
