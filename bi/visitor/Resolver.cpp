/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

bi::Resolver::Resolver() :
    memberScope(nullptr),
    currentClass(nullptr) {
  //
}

bi::Resolver::~Resolver() {
  //
}

bi::File* bi::Resolver::modify(File* o) {
  scopes.push_back(o->scope);
  o->root = o->root->accept(this);
  scopes.pop_back();
  return o;
}

bi::Expression* bi::Resolver::modify(List<Expression>* o) {
  Modifier::modify(o);
  o->type = new ListType(o->head->type->accept(&cloner),
      o->tail->type->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Parentheses* o) {
  Modifier::modify(o);
  o->type = new TupleType(o->single->type->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Binary* o) {
  Modifier::modify(o);
  o->type = new BinaryType(o->left->type->accept(&cloner),
      o->right->type->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  return o;
}

bi::Type* bi::Resolver::modify(IdentifierType* o) {
  return lookup(o)->accept(this);
}

bi::Type* bi::Resolver::modify(ClassType* o) {
  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Type* bi::Resolver::modify(AliasType* o) {
  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Type* bi::Resolver::modify(BasicType* o) {
  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Expression* bi::Resolver::lookup(Identifier<Unknown>* ref, Scope* scope) {
  LookupResult category = UNRESOLVED;
  if (scope) {
    /* use provided scope, usually a membership scope */
    category = scope->lookup(ref);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(ref);
    }
  }

  /* replace the reference of unknown object type with that of a known one */
  switch (category) {
  case PARAMETER:
    return new Identifier<Parameter>(ref->name, ref->loc);
  case MEMBER_PARAMETER:
    return new Identifier<MemberParameter>(ref->name, ref->loc);
  case GLOBAL_VARIABLE:
    return new Identifier<GlobalVariable>(ref->name, ref->loc);
  case LOCAL_VARIABLE:
    return new Identifier<LocalVariable>(ref->name, ref->loc);
  case MEMBER_VARIABLE:
    return new Identifier<MemberVariable>(ref->name, ref->loc);
  case FUNCTION:
    return new OverloadedIdentifier<Function>(ref->name, ref->loc);
  case FIBER:
    return new OverloadedIdentifier<Fiber>(ref->name, ref->loc);
  case MEMBER_FUNCTION:
    return new OverloadedIdentifier<MemberFunction>(ref->name, ref->loc);
  case MEMBER_FIBER:
    return new OverloadedIdentifier<MemberFiber>(ref->name, ref->loc);
  default:
    throw UnresolvedException(ref);
  }
}

bi::Type* bi::Resolver::lookup(IdentifierType* ref, Scope* scope) {
  LookupResult category = UNRESOLVED;
  if (scope) {
    /* use provided scope, usually a membership scope */
    category = scope->lookup(ref);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(ref);
    }
  }

  /* replace the reference of unknown object type with that of a known one */
  switch (category) {
  case BASIC:
    return new BasicType(ref->name, ref->loc);
  case CLASS:
    return new ClassType(ref->name, ref->loc);
  case ALIAS:
    return new AliasType(ref->name, ref->loc);
  default:
    throw UnresolvedException(ref);
  }
}

void bi::Resolver::checkBoolean(const Expression* o) {
  static BasicType type(new Name("Boolean"));
  scopes.front()->resolve(&type);
  if (!o->type->definitely(type)) {
    throw ConditionException(o);
  }
}

void bi::Resolver::checkInteger(const Expression* o) {
  static BasicType type32(new Name("Integer32"));
  static BasicType type64(new Name("Integer64"));
  scopes.front()->resolve(&type32);
  scopes.front()->resolve(&type64);
  if (!o->type->definitely(type32) && !o->type->definitely(type64)) {
    throw IndexException(o);
  }
}
