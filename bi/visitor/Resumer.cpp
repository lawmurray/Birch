/**
 * @file
 */
#include "bi/visitor/Resumer.hpp"

bi::Resumer::Resumer(const Yield* yield) :
    yield(yield),
    foundYield(false),
    inLoop(0) {
  //
}

bi::Resumer::~Resumer() {
  //
}

bi::Statement* bi::Resumer::clone(const Yield* o) {
  if (o == yield) {
    foundYield = true;
  }
  return Cloner::clone(o);
}

bi::Statement* bi::Resumer::clone(const Fiber* o) {
  return new Function(o->annotation, o->name, o->typeParams->accept(this),
      o->params->accept(this), o->returnType->accept(this),
      o->braces->accept(this), o->loc);
}

bi::Statement* bi::Resumer::clone(const If* o) {
  auto cond = o->cond->accept(this);
  auto foundBefore = foundYield;
  auto trueBraces = o->braces->accept(this);
  auto foundTrue = foundYield;
  auto falseBraces = o->falseBraces->accept(this);
  auto foundFalse = foundYield;

  if (inLoop || foundBefore) {
    return new If(cond, trueBraces, falseBraces, o->loc);
  } else if (foundTrue) {
    return trueBraces;
  } else if (foundFalse) {
    return falseBraces;
  } else {
    return new EmptyStatement(o->loc);
  }
}

bi::Statement* bi::Resumer::clone(const StatementList* o) {
  auto foundBefore = foundYield;
  auto head = o->head->accept(this);
  auto foundHead = foundYield;
  auto tail = o->tail->accept(this);
  auto foundTail = foundYield;

  auto keepHead = inLoop || foundBefore || foundHead || head->isDeclaration();
  auto keepTail = inLoop || foundBefore || foundTail || tail->isDeclaration();

  if (keepHead && keepTail) {
    return new StatementList(head, tail, o->loc);
  } else if (keepHead) {
    return head;
  } else if (keepTail) {
    return tail;
  } else {
    return new EmptyStatement(o->loc);
  }
}

bi::Statement* bi::Resumer::clone(const While* o) {
  ++inLoop;
  auto r = Cloner::clone(o);
  --inLoop;
  return r;
}

bi::Statement* bi::Resumer::clone(const DoWhile* o) {
  ++inLoop;
  auto r = Cloner::clone(o);
  --inLoop;
  return r;
}
