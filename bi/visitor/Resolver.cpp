/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/exception/all.hpp"

bi::Resolver::Resolver(shared_ptr<Scope> scope) :
    inInputs(false), inRandom(false) {
  if (scope) {
    push(scope);
  }
}

bi::Resolver::~Resolver() {
  //
}

void bi::Resolver::modify(File* o) {
  if (o->state == File::RESOLVING) {
    throw CyclicImportException(o);
  } else if (o->state == File::UNRESOLVED) {
    o->state = File::RESOLVING;
    o->scope = new Scope();
    files.push(o);
    push(o->scope);
    o->imports = o->imports->acceptModify(this);
    o->root = o->root->acceptModify(this);
    undefer();
    pop();
    files.pop();
    o->state = File::RESOLVED;
  }
}

bi::Expression* bi::Resolver::modify(ExpressionList* o) {
  Modifier::modify(o);
  o->type = new TypeList(o->head->type->acceptClone(&cloner),
      o->tail->type->acceptClone(&cloner));
  o->type = o->type->acceptModify(this);
  return o;
}

bi::Expression* bi::Resolver::modify(ParenthesesExpression* o) {
  Modifier::modify(o);
  o->type = new ParenthesesType(o->expr->type->acceptClone(&cloner));
  o->type = o->type->acceptModify(this);
  return o;
}

bi::Expression* bi::Resolver::modify(Range* o) {
  Modifier::modify(o);
  return o;
}

bi::Expression* bi::Resolver::modify(Traversal* o) {
  o->left = o->left->acceptModify(this);
  ModelReference* type = dynamic_cast<ModelReference*>(o->left->type.get());
  if (!type) {
    throw TraversalException(o);
  }
  traverseScope = type->target->scope;
  if (!traverseScope) {
    throw TraversalException(o);
  }
  o->right = o->right->acceptModify(this);
  o->type = o->right->type->acceptClone(&cloner)->acceptModify(this);
  return o;
}

bi::Expression* bi::Resolver::modify(This* o) {
  if (!model()) {
    throw ThisException(o);
  } else {
    Modifier::modify(o);
    o->type = new ModelReference(model()->name, new EmptyExpression(),
        nullptr, model());
  }
  return o;
}

bi::Expression* bi::Resolver::modify(BracketsExpression* o) {
  Modifier::modify(o);

  ModelReference* ref = dynamic_cast<ModelReference*>(o->expr->type.get());
  assert(ref);  ///@todo Exception

  const int typeSize = ref->ndims;
  const int indexSize = o->brackets->tupleSize();
  const int indexDims = o->brackets->tupleDims();

  assert(typeSize == indexSize);  ///@todo Exception
  ref = new ModelReference(ref->name, indexDims);

  o->type = ref->acceptModify(this);

  return o;
}

bi::Expression* bi::Resolver::modify(VarReference* o) {
  shared_ptr<Scope> scope = inner();
  Modifier::modify(o);
  o->target = scope->resolve(o);
  o->type = o->target->type->acceptClone(&cloner)->acceptModify(this);
  o->type->assignable = o->target->type->assignable;

  /* substitute with random variable if possible */
  RandomReference* random = new RandomReference(o);
  try {
    random->acceptModify(this);
    return random;
  } catch (UnresolvedReferenceException e) {
    delete random;
    return o;
  }
}

bi::Expression* bi::Resolver::modify(FuncReference* o) {
  shared_ptr<Scope> scope = inner();
  Modifier::modify(o);
  o->target = scope->resolve(o);
  o->type = o->target->type->acceptClone(&cloner)->acceptModify(this);
  o->form = o->target->form;

  /* substitute with random variable if possible */
  if (!inInputs && !inRandom && (*o->name == "<~" || *o->name == "~>")) {
    inRandom = true;
    RandomParameter* random = new RandomParameter(
        o->getLeft()->acceptClone(&cloner), new Name("~"),
        o->getRight()->acceptClone(&cloner), o->loc);
    o->name = new Name("<~");
    random->pull = o->acceptClone(&cloner);
    o->name = new Name("~>");
    random->push = o->acceptClone(&cloner);
    random->acceptModify(this);
    inRandom = false;
    return random;
  } else {
    if (o->isAssignment()) {
      if (inInputs) {
        o->getLeft()->type->assignable = true;
      } else if (!o->getLeft()->type->assignable) {
        throw NotAssignable(o);
      }
    }

    Gatherer<VarParameter> gatherer;
    o->target->parens->accept(&gatherer);
    for (auto iter = gatherer.gathered.begin();
        iter != gatherer.gathered.end(); ++iter) {
      o->args.push_back((*iter)->arg);
    }

    return o;
  }
}

bi::Expression* bi::Resolver::modify(RandomReference* o) {
  shared_ptr<Scope> scope = inner();
  Modifier::modify(o);
  o->target = scope->resolve(o);
  o->type = o->target->type->acceptClone(&cloner)->acceptModify(this);
  o->type->assignable = true;
  return o;
}

bi::Type* bi::Resolver::modify(ModelReference* o) {
  shared_ptr<Scope> scope = inner();
  Modifier::modify(o);
  o->target = scope->resolve(o);
  return o;
}

bi::Expression* bi::Resolver::modify(VarParameter* o) {
  Modifier::modify(o);
  if (!inInputs) {
    o->type->assignable = true;
  }
  if (*o->name) {
    inner()->add(o);
  }
  return o;
}

bi::Expression* bi::Resolver::modify(FuncParameter* o) {
  push();
  inInputs = true;
  o->parens = o->parens->acceptModify(this);
  inInputs = false;
  o->result = o->result->acceptModify(this);
  o->type = o->result->type->acceptClone(&cloner)->acceptModify(this);
  defer(o->braces.get());
  o->scope = pop();
  inner()->add(o);

  if (o->isAssignment()) {
    o->getLeft()->type->assignable = true;
  }

  Gatherer<VarParameter> gatherer1;
  o->parens->accept(&gatherer1);
  o->inputs = gatherer1.gathered;

  Gatherer<VarParameter> gatherer2;
  o->result->accept(&gatherer2);
  o->outputs = gatherer2.gathered;

  return o;
}

bi::Expression* bi::Resolver::modify(RandomParameter* o) {
  Modifier::modify(o);
  o->type = new RandomType(o->left->type->acceptClone(&cloner),
      o->right->type->acceptClone(&cloner));
  inner()->add(o);
  return o;
}

bi::Prog* bi::Resolver::modify(ProgParameter* o) {
  push();
  o->parens = o->parens->acceptModify(this);
  defer(o->braces.get());
  o->scope = pop();
  inner()->add(o);

  Gatherer<VarParameter> gatherer1;
  o->parens->accept(&gatherer1);
  o->inputs = gatherer1.gathered;

  return o;
}

bi::Type* bi::Resolver::modify(ModelParameter* o) {
  push();
  o->parens = o->parens->acceptModify(this);
  o->base = o->base->acceptModify(this);
  models.push(o);
  o->braces = o->braces->acceptModify(this);
  models.pop();
  o->scope = pop();
  inner()->add(o);

  if (*o->op != "=") {
    /* create constructor */
    Expression* parens1 = o->parens->acceptClone(&cloner);
    VarParameter* result1 = new VarParameter(new Name(),
        new ModelReference(o->name, 0, o));
    o->constructor = new FuncParameter(o->name, parens1, result1,
        new EmptyExpression(), CONSTRUCTOR);
    o->constructor =
        dynamic_cast<FuncParameter*>(o->constructor->acceptModify(this));
    assert(o->constructor);

    /* create assignment operator */
    Expression* right = new VarParameter(new Name(),
        new ModelReference(o->name, 0, o));
    Expression* left = new VarParameter(new Name(),
        new ModelReference(o->name, 0, o));
    Expression* parens2 = new ParenthesesExpression(
        new ExpressionList(left, right));
    Expression* result2 = new VarParameter(new Name(),
        new ModelReference(o->name, 0, o));
    o->assignment = new FuncParameter(new Name("<-"), parens2, result2,
        new EmptyExpression(), ASSIGNMENT_OPERATOR);
    o->assignment = dynamic_cast<FuncParameter*>(o->assignment->acceptModify(
        this));
    assert(o->assignment);
  }

  return o;
}

bi::Statement* bi::Resolver::modify(Import* o) {
  o->file->acceptModify(this);
  inner()->import(o->file->scope);
  return o;
}

bi::Statement* bi::Resolver::modify(VarDeclaration* o) {
  Modifier::modify(o);
  o->param->type->assignable = true;
  return o;
}

bi::Statement* bi::Resolver::modify(Conditional* o) {
  push();
  Modifier::modify(o);
  o->scope = pop();
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::Statement* bi::Resolver::modify(Loop* o) {
  push();
  Modifier::modify(o);
  o->scope = pop();
  ///@todo Check that condition is of type Boolean
  return o;
}

bi::shared_ptr<bi::Scope> bi::Resolver::inner() {
  shared_ptr<Scope> scope = nullptr;
  if (traverseScope) {
    scope = traverseScope;
    traverseScope = nullptr;
  } else if (scopes.size() > 0) {
    scope = scopes.top();
  }

  /* post-condition */
  assert(scope);

  return scope;
}

void bi::Resolver::push(shared_ptr<Scope> scope) {
  if (scope) {
    scopes.push(scope);
  } else {
    scopes.push(new Scope(inner()));
  }
}

bi::shared_ptr<bi::Scope> bi::Resolver::pop() {
  /* pre-conditions */
  assert(scopes.size() > 0);

  shared_ptr<Scope> res = scopes.top();
  scopes.pop();
  return res;
}

void bi::Resolver::defer(Expression* o) {
  if (files.size() == 1) {
    /* can ignore bodies in imported files */
    defers.push_back(std::make_tuple(o, inner(), model()));
  }
}

void bi::Resolver::undefer() {
  if (files.size() == 1) {
    auto iter = defers.begin();
    while (iter != defers.end()) {
      auto o = std::get<0>(*iter);
      auto scope = std::get<1>(*iter);
      auto model = std::get<2>(*iter);

      push(scope);
      models.push(model);
      o->acceptModify(this);
      models.pop();
      pop();
      ++iter;
    }
    defers.clear();
  }
}

bi::ModelParameter* bi::Resolver::model() {
  if (models.empty()) {
    return nullptr;
  } else {
    return models.top();
  }
}
