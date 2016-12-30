/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/exception/all.hpp"

bi::Resolver::Resolver(shared_ptr<Scope> scope) :
    inInputs(false) {
  if (scope) {
    push(scope);
  }
}

bi::Resolver::~Resolver() {
  //
}

void bi::Resolver::modify(File* file) {
  if (file->state == File::RESOLVING) {
    throw CyclicImportException(file);
  } else if (file->state == File::UNRESOLVED) {
    file->state = File::RESOLVING;
    file->scope = new Scope();
    files.push(file);
    push(file->scope);
    file->imports->acceptModify(this);
    file->root->acceptModify(this);
    undefer();
    pop();
    files.pop();
    file->state = File::RESOLVED;
  }
}

void bi::Resolver::modify(ExpressionList* o) {
  Modifier::modify(o);
  o->type = new TypeList(o->head->type->acceptClone(&cloner),
      o->tail->type->acceptClone(&cloner));
  o->type->acceptModify(this);
}

void bi::Resolver::modify(ParenthesesExpression* o) {
  Modifier::modify(o);
  o->type = new ParenthesesType(o->expr->type->acceptClone(&cloner));
  o->type->acceptModify(this);
}

void bi::Resolver::modify(BracesExpression* o) {
  Modifier::modify(o);
}

void bi::Resolver::modify(RandomVariable* o) {
  Modifier::modify(o);
  o->type = new RandomVariableType(o->left->type->acceptClone(&cloner),
      o->right->type->acceptClone(&cloner));
}

void bi::Resolver::modify(Range* o) {
  Modifier::modify(o);
}

void bi::Resolver::modify(Traversal* o) {
  o->left->acceptModify(this);
  ModelReference* type = dynamic_cast<ModelReference*>(o->left->type.get());
  if (!type) {
    throw TraversalException(o);
  }
  traverseScope = type->target->scope;
  if (!traverseScope) {
    throw TraversalException(o);
  }
  o->right->acceptModify(this);
  o->type = o->right->type->acceptClone(&cloner);
  o->type->acceptModify(this);
}

void bi::Resolver::modify(This* o) {
  if (!model()) {
    throw ThisException(o);
  } else {
    Modifier::modify(o);
    o->type = new ModelReference(model()->name, new EmptyExpression(),
        nullptr, model());
  }
}

void bi::Resolver::modify(BracketsExpression* o) {
  Modifier::modify(o);

  ModelReference* ref = dynamic_cast<ModelReference*>(o->expr->type.get());
  assert(ref);  ///@todo Exception

  const int typeSize = ref->ndims;
  const int indexSize = o->brackets->tupleSize();
  const int indexDims = o->brackets->tupleDims();

  assert(typeSize == indexSize);  ///@todo Exception
  ref = new ModelReference(ref->name, indexDims);

  o->type = ref;
  o->type->acceptModify(this);
}

void bi::Resolver::modify(VarReference* o) {
  shared_ptr<Scope> scope = inner();
  Modifier::modify(o);
  o->target = scope->resolve(o);
  o->type = o->target->type->acceptClone(&cloner);
  o->type->acceptModify(this);
  o->type->assignable = o->target->type->assignable;
}

void bi::Resolver::modify(FuncReference* o) {
  shared_ptr<Scope> scope = inner();
  Modifier::modify(o);
  o->target = scope->resolve(o);
  o->type = o->target->type->acceptClone(&cloner);
  o->type->acceptModify(this);
  o->form = o->target->form;

  if (o->isAssignment()) {
    if (inInputs) {
      o->getLeft()->type->assignable = true;
    } else if (!o->getLeft()->type->assignable) {
      throw NotAssignable(o);
    }
  }

  Gatherer<VarParameter> gatherer;
  o->target->parens->accept(&gatherer);
  for (auto iter = gatherer.gathered.begin(); iter != gatherer.gathered.end();
      ++iter) {
    o->args.push_back((*iter)->arg);
  }
}

void bi::Resolver::modify(ModelReference* o) {
  shared_ptr<Scope> scope = inner();
  Modifier::modify(o);
  o->target = scope->resolve(o);
}

void bi::Resolver::modify(VarParameter* o) {
  Modifier::modify(o);
  if (!inInputs) {
    o->type->assignable = true;
  }
  if (*o->name) {
    inner()->add(o);
  }
}

void bi::Resolver::modify(FuncParameter* o) {
  push();
  inInputs = true;
  o->parens->acceptModify(this);
  inInputs = false;
  o->result->acceptModify(this);
  o->type = o->result->type->acceptClone(&cloner);
  o->type->acceptModify(this);
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
}

void bi::Resolver::modify(ProgParameter* o) {
  push();
  o->parens->acceptModify(this);
  defer(o->braces.get());
  o->scope = pop();
  inner()->add(o);

  Gatherer<VarParameter> gatherer1;
  o->parens->accept(&gatherer1);
  o->inputs = gatherer1.gathered;
}

void bi::Resolver::modify(ModelParameter* o) {
  push();
  o->parens->acceptModify(this);
  o->base->acceptModify(this);
  models.push(o);
  o->braces->acceptModify(this);
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
    o->constructor->acceptModify(this);

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
    o->assignment->acceptModify(this);
  }
}

void bi::Resolver::modify(Import* o) {
  o->file->acceptModify(this);
  inner()->import(o->file->scope);
}

void bi::Resolver::modify(ExpressionStatement* o) {
  o->expr->acceptModify(this);
}

void bi::Resolver::modify(VarDeclaration* o) {
  o->param->acceptModify(this);
  o->param->type->assignable = true;
}

void bi::Resolver::modify(Conditional* o) {
  push();
  o->cond->acceptModify(this);
  o->braces->acceptModify(this);
  o->falseBraces->acceptModify(this);
  o->scope = pop();
  ///@todo Check that condition is of type Boolean
}

void bi::Resolver::modify(Loop* o) {
  push();
  o->cond->acceptModify(this);
  o->braces->acceptModify(this);
  o->scope = pop();
  ///@todo Check that condition is of type Boolean
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
