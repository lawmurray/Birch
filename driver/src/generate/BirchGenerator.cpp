/**
 * @file
 */
#include "src/generate/BirchGenerator.hpp"

#include "src/visitor/Gatherer.hpp"

birch::BirchGenerator::BirchGenerator(std::ostream& base, const int level,
    const bool header) :
    IndentableGenerator(base, level, header),
    type(nullptr) {
  base << std::fixed;
  // ^ forces floating point representation of integers to have decimal
  //   places
}

void birch::BirchGenerator::visit(const Package* o) {
  for (auto source : o->sources) {
    source->accept(this);
  }
  line("");
}

void birch::BirchGenerator::visit(const Name* o) {
  middle(o->str());
}

void birch::BirchGenerator::visit(const ExpressionList* o) {
  middle(o->head << ", " << o->tail);
}

void birch::BirchGenerator::visit(const Literal<bool>* o) {
  middle(o->str);
}

void birch::BirchGenerator::visit(const Literal<int64_t>* o) {
  middle(o->str);
}

void birch::BirchGenerator::visit(const Literal<double>* o) {
  middle(o->str);
}

void birch::BirchGenerator::visit(const Literal<const char*>* o) {
  middle(o->str);
}

void birch::BirchGenerator::visit(const Parentheses* o) {
  middle('(' << o->single << ')');
}

void birch::BirchGenerator::visit(const Sequence* o) {
  middle('[' << o->single << ']');
}

void birch::BirchGenerator::visit(const Cast* o) {
  middle(o->returnType << '?' << '(' << o->single << ')');
}

void birch::BirchGenerator::visit(const Call* o) {
  middle(o->single << '(' << o->args << ')');
}

void birch::BirchGenerator::visit(const BinaryCall* o) {
  middle(o->left << ' ' << o->name << ' ' << o->right);
}

void birch::BirchGenerator::visit(const UnaryCall* o) {
  middle(o->name << o->single);
}

void birch::BirchGenerator::visit(const Assign* o) {
  middle(o->left << ' ' << o->name << ' ' << o->right);
}

void birch::BirchGenerator::visit(const Slice* o) {
  middle(o->single << '[' << o->brackets << ']');
}

void birch::BirchGenerator::visit(const Query* o) {
  middle(o->single << '?');
}

void birch::BirchGenerator::visit(const Get* o) {
  middle(o->single << '!');
}

void birch::BirchGenerator::visit(const LambdaFunction* o) {
  middle("\\(" << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  middle(o->braces);
}

void birch::BirchGenerator::visit(const Span* o) {
  if (o->single->isEmpty()) {
    middle('_');
  } else {
    middle(o->single);
  }
}

void birch::BirchGenerator::visit(const Index* o) {
  middle(o->single);
}

void birch::BirchGenerator::visit(const Range* o) {
  middle(o->left << ".." << o->right);
}

void birch::BirchGenerator::visit(const Member* o) {
  middle(o->left << '.' << o->right);
}

void birch::BirchGenerator::visit(const Global* o) {
  middle("global." << o->single);
}

void birch::BirchGenerator::visit(const Super* o) {
  middle("super");
}

void birch::BirchGenerator::visit(const This* o) {
  middle("this");
}

void birch::BirchGenerator::visit(const Nil* o) {
  middle("nil");
}

void birch::BirchGenerator::visit(const Parameter* o) {
  middle(o->name << ':' << o->type);
  if (!o->value->isEmpty()) {
    middle(" <- " << o->value);
  }
}

void birch::BirchGenerator::visit(const Generic* o) {
  if (!o->type->isEmpty()) {
    middle(o->type);
  } else {
    middle(o->name);
  }
}

void birch::BirchGenerator::visit(const GlobalVariable* o) {
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

void birch::BirchGenerator::visit(const MemberVariable* o) {
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

void birch::BirchGenerator::visit(const LocalVariable* o) {
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

void birch::BirchGenerator::visit(const NamedExpression* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void birch::BirchGenerator::visit(const Braces* o) {
  finish(" {");
  in();
  middle(o->single);
  out();
  start('}');
}

void birch::BirchGenerator::visit(const Assume* o) {
  line(o->left << ' ' << o->name << ' ' << o->right << ';');
}

void birch::BirchGenerator::visit(const Function* o) {
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

void birch::BirchGenerator::visit(const Program* o) {
  start("program " << o->name << '(' << o->params << ')');
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void birch::BirchGenerator::visit(const MemberFunction* o) {
  start("");
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  if (o->has(OVERRIDE)) {
    middle("override ");
  }
  middle("function " << o->name);
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

void birch::BirchGenerator::visit(const BinaryOperator* o) {
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

void birch::BirchGenerator::visit(const UnaryOperator* o) {
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

void birch::BirchGenerator::visit(const AssignmentOperator* o) {
  start("operator <- " << o->single);
  if (!o->braces->isEmpty() && (!header || (type && type->isGeneric()))) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void birch::BirchGenerator::visit(const ConversionOperator* o) {
  start("operator -> " << o->returnType);
  if (!o->braces->isEmpty() && (!header || (type && type->isGeneric()))) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void birch::BirchGenerator::visit(const Class* o) {
  type = o;
  start("");
  if (o->has(ABSTRACT)) {
    middle("abstract ");
  }
  if (o->has(FINAL)) {
    middle("final ");
  }
  middle("class " << o->name);
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

void birch::BirchGenerator::visit(const Basic* o) {
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

void birch::BirchGenerator::visit(const ExpressionStatement* o) {
  line(o->single << ';');
}

void birch::BirchGenerator::visit(const If* o) {
  start("if " << o->cond << o->braces);
  if (!o->falseBraces->isEmpty()) {
    middle(" else" << o->falseBraces);
  }
  finish("");
}

void birch::BirchGenerator::visit(const For* o) {
  auto index = dynamic_cast<const LocalVariable*>(o->index);
  assert(index);
  start("for " << index->name << " in " << o->from << ".." << o->to);
  finish(o->braces);
}

void birch::BirchGenerator::visit(const Parallel* o) {
  auto index = dynamic_cast<const LocalVariable*>(o->index);
  assert(index);
  start("");
  if (o->has(DYNAMIC)) {
    middle("dynamic ");
  }
  middle("parallel for " << index->name << " in " << o->from << ".." << o->to);
  finish(o->braces);
}

void birch::BirchGenerator::visit(const While* o) {
  line("while " << o->cond << o->braces);
}

void birch::BirchGenerator::visit(const DoWhile* o) {
  line("do " << o->braces << " while " << o->cond << ';');
}

void birch::BirchGenerator::visit(const With* o) {
  line("with " << o->single << o->braces);
}

void birch::BirchGenerator::visit(const Assert* o) {
  line("assert " << o->cond << ';');
}

void birch::BirchGenerator::visit(const Return* o) {
  line("return " << o->single << ';');
}

void birch::BirchGenerator::visit(const Factor* o) {
  line("factor " << o->single << ';');
}

void birch::BirchGenerator::visit(const Raw* o) {
  line(o->name << "{{");
  start(o->raw);
  finish("}}");
}

void birch::BirchGenerator::visit(const StatementList* o) {
  middle(o->head << o->tail);
}

void birch::BirchGenerator::visit(const NamedType* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void birch::BirchGenerator::visit(const MemberType* o) {
  middle(o->left << '.' << o->right);
}

void birch::BirchGenerator::visit(const ArrayType* o) {
  middle(o->single << '[');
  for (int i = 0; i < o->depth(); ++i) {
    if (i > 0) {
      middle(',');
    }
    middle('_');
  }
  middle(']');
}

void birch::BirchGenerator::visit(const TupleType* o) {
  middle('(' << o->single << ')');
}

void birch::BirchGenerator::visit(const FunctionType* o) {
  middle("\\(" << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void birch::BirchGenerator::visit(const OptionalType* o) {
  middle(o->single << '?');
}

void birch::BirchGenerator::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}
