/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/visitor/Instantiater.hpp"

bi::Resolver::Resolver(Scope* rootScope) {
  scopes.push_back(rootScope);
}

bi::Resolver::~Resolver() {
  //
}

bi::Expression* bi::Resolver::modify(ExpressionList* o) {
  Modifier::modify(o);
  o->type = new TypeList(o->head->type, o->tail->type, o->loc,
      o->head->type->assignable && o->tail->type->assignable);
  return o;
}

bi::Expression* bi::Resolver::modify(Parentheses* o) {
  Modifier::modify(o);
  if (o->single->count() > 1) {
    o->type = new TupleType(o->single->type, o->loc,
        o->single->type->assignable);
  } else {
    o->type = o->single->type;
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Binary* o) {
  Modifier::modify(o);
  o->type = new BinaryType(o->left->type, o->right->type, o->loc);
  return o;
}

bi::Type* bi::Resolver::modify(TypeIdentifier* o) {
  return lookup(o)->accept(this);
}

bi::Type* bi::Resolver::modify(ClassType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o);
  if (!o->typeArgs->isEmpty() || o->target->isGeneric()) {
    if (o->typeArgs->count() == o->target->typeParams->count()) {
      Class* instantiation = o->target->getInstantiation(o->typeArgs);
      if (!instantiation) {
        Instantiater instantiater(o->typeArgs);
        instantiation =
            dynamic_cast<Class*>(o->target->accept(&instantiater));
        assert(instantiation);
        o->target->addInstantiation(instantiation);
        instantiation->accept(this);
      }
      o->target = instantiation;
    } else {
      throw GenericException(o, o->target);
    }
  }
  return o;
}

bi::Type* bi::Resolver::modify(AliasType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Type* bi::Resolver::modify(BasicType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Type* bi::Resolver::modify(GenericType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o);
  return o;
}

bi::Expression* bi::Resolver::lookup(Identifier<Unknown>* ref) {
  LookupResult category = UNRESOLVED;
  if (!memberScopes.empty()) {
    /* use membership scope */
    category = memberScopes.back()->lookup(ref);
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

bi::Type* bi::Resolver::lookup(TypeIdentifier* ref) {
  LookupResult category = UNRESOLVED;
  if (!memberScopes.empty()) {
    /* use membership scope */
    category = memberScopes.back()->lookup(ref);
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
  case GENERIC:
    return new GenericType(ref->name, ref->loc);
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
