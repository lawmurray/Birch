/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

bi::Resolver::Resolver(Scope* rootScope, const bool pointers) :
    pointers(pointers) {
  scopes.push_back(rootScope);
}

bi::Resolver::~Resolver() {
  //
}

bi::Expression* bi::Resolver::modify(ExpressionList* o) {
  Modifier::modify(o);
  o->type = new TypeList(o->head->type, o->tail->type, o->loc);
  return o;
}

bi::Expression* bi::Resolver::modify(Parentheses* o) {
  Modifier::modify(o);
  if (o->single->width() > 1) {
    o->type = new TupleType(o->single->type, o->loc);
  } else {
    o->type = o->single->type;
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Sequence* o) {
  Modifier::modify(o);
  auto iter = o->single->type->begin();
  if (iter == o->single->type->end()) {
    o->type = new NilType(o->loc);
  } else {
    auto common = *iter;
    ++iter;
    while (common && iter != o->single->type->end()) {
      common = common->common(**iter);
      ++iter;
    }
    if (!common) {
      throw SequenceException(o);
    } else {
      o->type = new SequenceType(common, o->loc);
    }
  }
  return o;
}

bi::Expression* bi::Resolver::modify(Binary* o) {
  Modifier::modify(o);
  o->type = new BinaryType(o->left->type, o->right->type, o->loc);
  return o;
}

bi::Type* bi::Resolver::modify(UnknownType* o) {
  return lookup(o)->accept(this);
}

bi::Type* bi::Resolver::modify(ClassType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  instantiate(o);
  return o;
}

bi::Type* bi::Resolver::modify(BasicType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o, GLOBAL_SCOPE);
  return o;
}

bi::Type* bi::Resolver::modify(GenericType* o) {
  assert(!o->target);
  Modifier::modify(o);
  resolve(o, CLASS_SCOPE);
  return o;
}

bi::Type* bi::Resolver::modify(MemberType* o) {
  o->left = o->left->accept(this);
  if (o->left->isClass()) {
    memberScopes.push_back(o->left->getClass()->scope);
    o->right = o->right->accept(this);
  } else if (o->left->isBound()) {
    throw MemberException(o);
  }
  return o;
}

bi::Expression* bi::Resolver::lookup(Identifier<Unknown>* o) {
  LookupResult category = UNRESOLVED;
  if (!memberScopes.empty()) {
    /* use membership scope */
    category = memberScopes.back()->lookup(o);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(o);
    }
  }

  switch (category) {
  case PARAMETER:
    return new Identifier<Parameter>(o->name, o->loc);
  case GLOBAL_VARIABLE:
    return new Identifier<GlobalVariable>(o->name, o->loc);
  case LOCAL_VARIABLE:
    return new Identifier<LocalVariable>(o->name, o->loc);
  case MEMBER_VARIABLE:
    return new Identifier<MemberVariable>(o->name, o->loc);
  case FUNCTION:
    return new OverloadedIdentifier<Function>(o->name, new EmptyType(o->loc), o->loc);
  case FIBER:
    return new OverloadedIdentifier<Fiber>(o->name, new EmptyType(o->loc), o->loc);
  case MEMBER_FUNCTION:
    return new OverloadedIdentifier<MemberFunction>(o->name, new EmptyType(o->loc), o->loc);
  case MEMBER_FIBER:
    return new OverloadedIdentifier<MemberFiber>(o->name, new EmptyType(o->loc), o->loc);
  default:
    throw UnresolvedException(o);
  }
}

bi::Expression* bi::Resolver::lookup(OverloadedIdentifier<Unknown>* o) {
  LookupResult category = UNRESOLVED;
  if (!memberScopes.empty()) {
    /* use membership scope */
    category = memberScopes.back()->lookup(o);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(o);
    }
  }

  switch (category) {
  case FUNCTION:
    return new OverloadedIdentifier<Function>(o->name, o->typeArgs, o->loc);
  case FIBER:
    return new OverloadedIdentifier<Fiber>(o->name, o->typeArgs, o->loc);
  case MEMBER_FUNCTION:
    return new OverloadedIdentifier<MemberFunction>(o->name, o->typeArgs, o->loc);
  case MEMBER_FIBER:
    return new OverloadedIdentifier<MemberFiber>(o->name, o->typeArgs, o->loc);
  default:
    throw UnresolvedException(o);
  }
}

bi::Type* bi::Resolver::lookup(UnknownType* o) {
  LookupResult category = UNRESOLVED;
  if (!memberScopes.empty()) {
    /* use membership scope */
    category = memberScopes.back()->lookup(o);
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin();
        category == UNRESOLVED && iter != scopes.rend(); ++iter) {
      category = (*iter)->lookup(o);
    }
  }

  switch (category) {
  case BASIC:
    return new BasicType(o->name, o->loc);
  case CLASS:
    return new PointerType(o->weak,
        new ClassType(o->name, o->typeArgs, o->loc), o->loc);
  case GENERIC:
    return new GenericType(o->name, o->loc);
  default:
    throw UnresolvedException(o);
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
  static BasicType type(new Name("Integer"));
  scopes.front()->resolve(&type);
  if (!o->type->definitely(type)) {
    throw IndexException(o);
  }
}
