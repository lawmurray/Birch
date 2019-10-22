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

  /* generic function and fiber instantiations from dependencies */
  Gatherer<Class> headerClasses;
  Gatherer<Function> headerFunctions;
  Gatherer<Fiber> headerFibers;

  for (auto file : o->headers) {
    file->accept(&headerClasses);
    file->accept(&headerFunctions);
    file->accept(&headerFibers);
  }
  for (auto o : headerClasses) {
    for (auto instantiation : o->instantiations) {
      if (!instantiation->has(PRIOR_INSTANTIATION)) {
        *this << instantiation;
      }
    }
  }
  for (auto o : headerFunctions) {
    for (auto instantiation : o->instantiations) {
      if (!instantiation->has(PRIOR_INSTANTIATION)) {
        *this << instantiation;
      }
    }
  }
  for (auto o : headerFibers) {
    for (auto instantiation : o->instantiations) {
      if (!instantiation->has(PRIOR_INSTANTIATION)) {
        *this << instantiation;
      }
    }
  }
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

void bi::bi_ostream::visit(const Call<Unknown>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<Function>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<MemberFunction>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<Fiber>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<MemberFiber>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<Parameter>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<LocalVariable>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<MemberVariable>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<GlobalVariable>* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const Call<BinaryOperator>* o) {
  middle(
      o->args->getLeft() << ' ' << o->single << ' ' << o->args->getRight());
}

void bi::bi_ostream::visit(const Call<UnaryOperator>* o) {
  middle(o->single << o->args);
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

void bi::bi_ostream::visit(const LambdaFunction* o) {
  middle("@(" << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
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

void bi::bi_ostream::visit(const LocalVariable* o) {
  if (o->has(AUTO)) {
    middle("auto " << o->name);
  } else {
    middle(o->name << ':');
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

void bi::bi_ostream::visit(const Identifier<Parameter>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const Identifier<GlobalVariable>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const Identifier<LocalVariable>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const Identifier<MemberVariable>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const Identifier<Unknown>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const OverloadedIdentifier<Function>* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::bi_ostream::visit(const OverloadedIdentifier<Fiber>* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::bi_ostream::visit(const OverloadedIdentifier<MemberFunction>* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::bi_ostream::visit(const OverloadedIdentifier<MemberFiber>* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::bi_ostream::visit(const OverloadedIdentifier<BinaryOperator>* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::bi_ostream::visit(const OverloadedIdentifier<UnaryOperator>* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::bi_ostream::visit(const OverloadedIdentifier<Unknown>* o) {
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
  line(o->left << " ~ " << o->right << ';');
}

void bi::bi_ostream::visit(const Function* o) {
  if (o->isInstantiation() && !o->has(PRIOR_INSTANTIATION)) {
    line("instantiated function " << o->name << '<' << o->typeParams << ">;");
  } else {
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
  for (auto instantiation : o->instantiations) {
    instantiation->accept(this);
  }
}

void bi::bi_ostream::visit(const Fiber* o) {
  if (o->isInstantiation() && !o->has(PRIOR_INSTANTIATION)) {
    line("instantiated fiber " << o->name << '<' << o->typeParams << ">;");
  } else {
    start("fiber " << o->name);
    if (!o->typeParams->isEmpty()) {
      middle('<' << o->typeParams << '>');
    }
    middle('(' << o->params << ')');
    if (!o->returnType->unwrap()->isEmpty()) {
      middle(" -> " << o->returnType->unwrap());
    }
    if (!o->braces->isEmpty() && (!header || o->isGeneric())) {
      finish(o->braces << "\n");
    } else {
      finish(';');
    }
  }
  for (auto instantiation : o->instantiations) {
    instantiation->accept(this);
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
  start("function " << o->name << '(' << o->params << ')');
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
  start("fiber " << o->name << '(' << o->params << ')');
  if (!o->returnType->unwrap()->isEmpty()) {
    middle(" -> " << o->returnType->unwrap());
  }
  if (!o->braces->isEmpty() && (!header || (type && type->isGeneric()))) {
    finish(o->braces << "\n");
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const BinaryOperator* o) {
  start("operator (");
  middle(o->params->getLeft());
  middle(' ' << o->name << ' ');
  middle(o->params->getRight());
  middle(')');
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
  start("operator (" << o->name << o->params << ')');
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
  if (o->isInstantiation() && !o->has(PRIOR_INSTANTIATION)) {
    line("instantiated class " << o->name << '<' << o->typeParams << ">;");
  } else {
    type = o;
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

  for (auto instantiation : o->instantiations) {
    instantiation->accept(this);
  }
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
  start("for (" << o->index << " in " << o->from << ".." << o->to << ')');
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

void bi::bi_ostream::visit(const ClassType* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
  if (o->weak) {
    middle('&');
  }
}

void bi::bi_ostream::visit(const BasicType* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const GenericType* o) {
  if (o->target && !o->target->type->isEmpty()) {
    middle(o->target->type);
  } else {
    middle(o->name);
  }
}

void bi::bi_ostream::visit(const MemberType* o) {
  middle(o->left << '.' << o->right);
}

void bi::bi_ostream::visit(const BinaryType* o) {
  middle('(' << o->left << ", " << o->right << ')');
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
  middle("@(" << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::bi_ostream::visit(const FiberType* o) {
  middle(o->single << '!');
}

void bi::bi_ostream::visit(const OptionalType* o) {
  middle(o->single << '?');
}

void bi::bi_ostream::visit(const UnknownType* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
  if (o->weak) {
    middle('&');
  }
}

void bi::bi_ostream::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}
