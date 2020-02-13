/**
 * @file
 */
#include "bi/visitor/Scoper.hpp"

bi::Scoper::Scoper() {
  //
}

bi::Scoper::~Scoper() {
  //
}

bi::Expression* bi::Scoper::modify(Parameter* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(LocalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(MemberVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(GlobalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(MemberFunction* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Function* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(MemberFiber* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Fiber* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(BinaryOperator* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(UnaryOperator* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Program* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Basic* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Class* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Scoper::modify(Yield* o) {
  /* construct the yield state, this being the parameters and local variables
   * that must be preserved for execution to resume */
  assert(o->state.empty());
  for (auto iter1 = scopes.begin(); iter1 != scopes.end(); ++iter1) {
    auto& params = (*iter1)->parameters;
    for (auto pair : params) {
      auto param = pair.second;
      auto named = new NamedExpression(param->name, param->loc);
      named->category = PARAMETER;
      named->number = param->number;
      o->state.push_back(named);
    }

    auto& locals = (*iter1)->localVariables;
    for (auto pair : locals) {
      auto local = pair.second;
      auto named = new NamedExpression(local->name, local->loc);
      named->category = LOCAL_VARIABLE;
      named->number = local->number;
      o->state.push_back(named);
    }
  }

  /* resume function */
  if (o->resume) {
    o->resume->accept(this);
  }
  return ScopedModifier::modify(o);
}

bi::Expression* bi::Scoper::modify(Generic* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}
