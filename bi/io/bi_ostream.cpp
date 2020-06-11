/**
 * @file
 */
#include "bi/io/bi_ostream.hpp"

#include "bi/visitor/Gatherer.hpp"

bi::bi_ostream::bi_ostream(std::ostream& base, const int level,
    const bool header) :
    indentable_ostream(base, level, header),
    type(nullptr) {
  base << std::fixed;
  // ^ forces floating point representation of integers to have decimal
  //   places
}

void bi::bi_ostream::visit(const Package* o) {
  for (auto source : o->sources) {
    source->accept(this);
  }
  line("");
}

void bi::bi_ostream::visit(const Name* o) {
  middle(o->str());
}

void bi::bi_ostream::visit(const ExpressionList* o) {
  middle(o->head << ", " << o->tail);
}

void bi::bi_ostream::visit(const Literal<bool>* o) {
  middle(o->str);
}

void bi::bi_ostream::visit(const Literal<int64_t>* o) {
  middle(o->str);
}

void bi::bi_ostream::visit(const Literal<double>* o) {
  middle(o->str);
}

void bi::bi_ostream::visit(const Literal<const char*>* o) {
  middle(o->str);
}

void bi::bi_ostream::visit(const Parentheses* o) {
  middle('(' << o->single << ')');
}

void bi::bi_ostream::visit(const Sequence* o) {
  middle('[' << o->single << ']');
}

void bi::bi_ostream::visit(const Cast* o) {
  middle(o->returnType << '?' << '(' << o->single << ')');
}

void bi::bi_ostream::visit(const Call* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const BinaryCall* o) {
  middle(o->left << ' ' << o->name << ' ' << o->right);
}

void bi::bi_ostream::visit(const UnaryCall* o) {
  middle(o->name << o->single);
}

void bi::bi_ostream::visit(const Assign* o) {
  middle(o->left << ' ' << o->name << ' ' << o->right);
}

void bi::bi_ostream::visit(const Slice* o) {
  middle(o->single << '[' << o->brackets << ']');
}

void bi::bi_ostream::visit(const Query* o) {
  middle(o->single << '?');
}

void bi::bi_ostream::visit(const Get* o) {
  middle(o->single << '!');
}

void bi::bi_ostream::visit(const GetReturn* o) {
  middle(o->single << '%');
}

void bi::bi_ostream::visit(const Spin* o) {
  middle('@' << o->single);
}

void bi::bi_ostream::visit(const LambdaFunction* o) {
  middle("\\(" << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  middle(o->braces);
}

void bi::bi_ostream::visit(const Span* o) {
  if (o->single->isEmpty()) {
    middle('_');
  } else {
    middle(o->single);
  }
}

void bi::bi_ostream::visit(const Index* o) {
  middle(o->single);
}

void bi::bi_ostream::visit(const Range* o) {
  middle(o->left << ".." << o->right);
}

void bi::bi_ostream::visit(const Member* o) {
  middle(o->left << '.' << o->right);
}

void bi::bi_ostream::visit(const Global* o) {
  middle("global." << o->single);
}

void bi::bi_ostream::visit(const Super* o) {
  middle("super");
}

void bi::bi_ostream::visit(const This* o) {
  middle("this");
}

void bi::bi_ostream::visit(const Nil* o) {
  middle("nil");
}

void bi::bi_ostream::visit(const Parameter* o) {
  middle(o->name << ':' << o->type);
  if (!o->value->isEmpty()) {
    middle(" <- " << o->value);
  }
}

void bi::bi_ostream::visit(const Generic* o) {
  if (!o->type->isEmpty()) {
    middle(o->type);
  } else {
    middle(o->name);
  }
}

void bi::bi_ostream::visit(const GlobalVariable* o) {
  if (o->has(AUTO)) {
    start("auto " << o->name);
  } else {
    start(o->name << ':');
    if (o->type->isArray() && !o->brackets->isEmpty()) {
      middle(dynamic_cast<const ArrayType*>(o->type)->single);
    } else {
      middle(o->type);
    }
    if (!o->brackets->isEmpty()) {
      middle('[' << o->brackets << ']');
    }
    if (!o->args->isEmpty()) {
      middle('(' << o->args << ')');
    }
  }
  if (!o->value->isEmpty()) {
    middle(" <- " << o->value);
  }
  finish(';');
}

void bi::bi_ostream::visit(const MemberVariable* o) {
  if (o->has(AUTO)) {
    start("auto " << o->name);
  } else {
    start(o->name << ':');
    if (o->type->isArray() && !o->brackets->isEmpty()) {
      middle(dynamic_cast<const ArrayType*>(o->type)->single);
    } else {
      middle(o->type);
    }
    if (!o->brackets->isEmpty()) {
      middle('[' << o->brackets << ']');
    }
    if (!o->args->isEmpty()) {
      middle('(' << o->args << ')');
    }
  }
  if (!o->value->isEmpty()) {
    middle(" <- " << o->value);
  }
  finish(';');
}

void bi::bi_ostream::visit(const LocalVariable* o) {
  if (o->has(AUTO)) {
    start("auto " << o->name);
  } else {
    start(o->name << ':');
    if (o->type->isArray() && !o->brackets->isEmpty()) {
      middle(dynamic_cast<const ArrayType*>(o->type)->single);
    } else {
      middle(o->type);
    }
    if (!o->brackets->isEmpty()) {
      middle('[' << o->brackets << ']');
    }
    if (!o->args->isEmpty()) {
      middle('(' << o->args << ')');
    }
  }
  if (!o->value->isEmpty()) {
    middle(" <- " << o->value);
  }
  finish(';');
}

void bi::bi_ostream::visit(const NamedExpression* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::bi_ostream::visit(const Braces* o) {
  finish(" {");
  in();
  middle(o->single);
  out();
  start('}');
}

void bi::bi_ostream::visit(const Assume* o) {
  line(o->left << ' ' << o->name << ' ' << o->right << ';');
}

void bi::bi_ostream::visit(const Function* o) {
  start("function " << o->name);
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
  middle('(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!o->braces->isEmpty() && (!header || o->isGeneric())) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const Fiber* o) {
  start("fiber " << o->name);
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
  middle('(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    auto type = dynamic_cast<const FiberType*>(o->returnType);
    assert(type);
    middle(" -> ");
    if (!type->returnType->isEmpty()) {
      middle(type->returnType << '%');
    }
    middle(type->yieldType);
  }
  if (!o->braces->isEmpty() && (!header || o->isGeneric())) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const Program* o) {
  start("program " << o->name << '(' << o->params << ')');
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const MemberFunction* o) {
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  if (o->has(OVERRIDE)) {
    middle("override ");
  }
  start("function " << o->name);
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
  middle('(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!o->braces->isEmpty() && (!header || (type && type->isGeneric()))) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const MemberFiber* o) {
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  if (o->has(OVERRIDE)) {
    middle("override ");
  }
  start("fiber " << o->name);
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
  middle('(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    auto type = dynamic_cast<const FiberType*>(o->returnType);
    assert(type);
    middle(" -> ");
    if (!type->returnType->isEmpty()) {
      middle(type->returnType << '%');
    }
    middle(type->yieldType);
  }
  if (!o->braces->isEmpty() && (!header || (type && type->isGeneric()))) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const BinaryOperator* o) {
  start("operator (" << o->left << ' ' << o->name << ' ' << o->right << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const UnaryOperator* o) {
  start("operator (" << o->name << o->single << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const AssignmentOperator* o) {
  start("operator <- " << o->single);
  if (!o->braces->isEmpty() && (!header || (type && type->isGeneric()))) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const ConversionOperator* o) {
  start("operator -> " << o->returnType);
  if (!o->braces->isEmpty() && (!header || (type && type->isGeneric()))) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const Class* o) {
  type = o;
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  start("class " << o->name);
  if (o->isGeneric()) {
    middle('<' << o->typeParams << '>');
  }
  if (!o->isAlias() && !o->params->isEmpty()) {
    middle('(' << o->params << ')');
  }
  if (!o->base->isEmpty()) {
    if (o->isAlias()) {
      middle(" = ");
    } else {
      middle(" < ");
    }
    middle(o->base);
    if (!o->args->isEmpty()) {
      middle('(' << o->args << ')');
    }
  }
  if (!o->braces->isEmpty()) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
  type = nullptr;
}

void bi::bi_ostream::visit(const Basic* o) {
  start("type " << o->name);
  if (!o->base->isEmpty()) {
    if (o->isAlias()) {
      middle(" = ");
    } else {
      middle(" < ");
    }
    middle(o->base);
  }
  finish(';');
}

void bi::bi_ostream::visit(const ExpressionStatement* o) {
  line(o->single << ';');
}

void bi::bi_ostream::visit(const If* o) {
  start("if " << o->cond << o->braces);
  if (!o->falseBraces->isEmpty()) {
    middle(" else" << o->falseBraces);
  }
  finish("");
}

void bi::bi_ostream::visit(const For* o) {
  auto index = dynamic_cast<const LocalVariable*>(o->index);
  assert(index);
  start("for " << index->name << " in " << o->from << ".." << o->to);
  finish(o->braces);
}

void bi::bi_ostream::visit(const Parallel* o) {
  auto index = dynamic_cast<const LocalVariable*>(o->index);
  assert(index);
  start("");
  if (o->has(DYNAMIC)) {
    middle("dynamic ");
  }
  middle("parallel for " << index->name << " in " << o->from << ".." << o->to);
  finish(o->braces);
}

void bi::bi_ostream::visit(const While* o) {
  line("while " << o->cond << o->braces);
}

void bi::bi_ostream::visit(const DoWhile* o) {
  line("do " << o->braces << " while " << o->cond << ';');
}

void bi::bi_ostream::visit(const Assert* o) {
  line("assert " << o->cond << ';');
}

void bi::bi_ostream::visit(const Return* o) {
  line("return " << o->single << ';');
}

void bi::bi_ostream::visit(const Yield* o) {
  line("yield " << o->single << ';');
}

void bi::bi_ostream::visit(const Raw* o) {
  line(o->name << "{{");
  start(o->raw);
  finish("}}");
}

void bi::bi_ostream::visit(const StatementList* o) {
  middle(o->head << o->tail);
}

void bi::bi_ostream::visit(const NamedType* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
  if (o->weak) {
    middle('&');
  }
}

void bi::bi_ostream::visit(const MemberType* o) {
  middle(o->left << '.' << o->right);
}

void bi::bi_ostream::visit(const ArrayType* o) {
  middle(o->single << '[');
  for (int i = 0; i < o->depth(); ++i) {
    if (i > 0) {
      middle(',');
    }
    middle('_');
  }
  middle(']');
}

void bi::bi_ostream::visit(const TupleType* o) {
  middle('(' << o->single << ')');
}

void bi::bi_ostream::visit(const FunctionType* o) {
  middle("\\(" << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::bi_ostream::visit(const FiberType* o) {
  if (!o->returnType->isEmpty()) {
    middle(o->returnType << '%');
  }
  middle(o->yieldType << '!');
}

void bi::bi_ostream::visit(const OptionalType* o) {
  middle(o->single << '?');
}

void bi::bi_ostream::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}
