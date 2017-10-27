/**
 * @file
 */
#include "bi/io/bi_ostream.hpp"

bi::bi_ostream::bi_ostream(std::ostream& base, const int level,
    const bool header) :
    indentable_ostream(base, level, header) {
  base << std::fixed;
  // ^ forces floating point representation of integers to have decimal
  //   places
}

void bi::bi_ostream::visit(const Package* o) {
  for (auto source: o->sources) {
    source->accept(this);
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

void bi::bi_ostream::visit(const Cast* o) {
  middle(o->returnType << '?' << '(' << o->single << ')');
}

void bi::bi_ostream::visit(const Call* o) {
  middle(o->single << '(' << o->args << ')');
}

void bi::bi_ostream::visit(const BinaryCall* o) {
  middle(o->args->getLeft() << ' ' << o->single << ' ' << o->args->getRight());
}

void bi::bi_ostream::visit(const UnaryCall* o) {
  middle(o->single << o->args);
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
  middle(o->name << ':' << o->type);
  if (!o->args->isEmpty()) {
    middle('(' << o->args << ')');
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

void bi::bi_ostream::visit(const MemberParameter* o) {
  middle(o->name << ':' << o->type);
  if (!o->value->isEmpty()) {
    middle(" <- " << o->value);
  }
}

void bi::bi_ostream::visit(const Generic* o) {
  middle(o->name);
  if (!o->type->isEmpty()) {
    middle(" <= " << o->type);
  }
}

void bi::bi_ostream::visit(const GlobalVariable* o) {
  start(o->name << ':' << o->type);
  if (!o->args->isEmpty()) {
    middle('(' << o->args << ')');
  }
  if (!o->value->isEmpty()) {
    middle(" <- " << o->value);
  }
  finish(';');
}

void bi::bi_ostream::visit(const MemberVariable* o) {
  start(o->name << ':' << o->type);
  if (!o->args->isEmpty()) {
    middle('(' << o->args << ')');
  }
  if (!o->value->isEmpty()) {
    middle(" <- " << o->value);
  }
  finish(';');
}

void bi::bi_ostream::visit(const Identifier<Parameter>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const Identifier<MemberParameter>* o) {
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

void bi::bi_ostream::visit(const OverloadedIdentifier<Function>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const OverloadedIdentifier<Fiber>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const OverloadedIdentifier<MemberFunction>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const OverloadedIdentifier<MemberFiber>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const OverloadedIdentifier<BinaryOperator>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const OverloadedIdentifier<UnaryOperator>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const Identifier<Unknown>* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const Braces* o) {
  finish(" {");
  in();
  middle(o->single);
  out();
  line('}');
}

void bi::bi_ostream::visit(const Assignment* o) {
  line(o->left << ' ' << o->name << ' ' << o->right << ';');
}

void bi::bi_ostream::visit(const Function* o) {
  start("function " << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const Fiber* o) {
  start("fiber " << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const Program* o) {
  start("program " << o->name << '(' << o->params << ')');
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const MemberFunction* o) {
  start("function " << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const MemberFiber* o) {
  start("fiber " << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces);
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
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const UnaryOperator* o) {
  start("operator (" << o->name << ' ' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const AssignmentOperator* o) {
  start("operator " << o->name << ' ' << o->single);
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const ConversionOperator* o) {
  start("operator -> " << o->returnType);
  if (!header && !o->braces->isEmpty()) {
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const Class* o) {
  start("class " << o->name);
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
  if (!o->params->isEmpty()) {
    middle('(' << o->params << ')');
  }
  if (!o->base->isEmpty()) {
    middle(" < " << o->base);
    if (!o->args->isEmpty()) {
      middle('(' << o->args << ')');
    }
  }
  if (!o->braces->isEmpty()) {
    finish(o->braces);
  } else {
    finish(';');
  }
}

void bi::bi_ostream::visit(const Alias* o) {
  line("type " << o->name << " = " << o->base << ';');
}

void bi::bi_ostream::visit(const Basic* o) {
  start("type " << o->name);
  if (!o->base->isEmpty()) {
    middle(" < " << o->base);
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
  start("for ("  << o->index << " in " << o->from << ".." << o->to << ')');
  finish(o->braces);
}

void bi::bi_ostream::visit(const While* o) {
  line("while " << o->cond << o->braces);
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
  if (!header) {
    line(o->name << " {{");
    line(o->raw);
    line("}}");
  }
}

void bi::bi_ostream::visit(const StatementList* o) {
  middle(o->head << o->tail);
}

void bi::bi_ostream::visit(const ClassType* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void bi::bi_ostream::visit(const AliasType* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const BasicType* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const GenericType* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const BinaryType* o) {
  middle('(' << o->left << ", " << o->right << ')');
}

void bi::bi_ostream::visit(const ArrayType* o) {
  middle(o->single << '[' << o->brackets << ']');
}

void bi::bi_ostream::visit(const TupleType* o) {
  middle('(' << o->single << ')');
}

void bi::bi_ostream::visit(const FunctionType* o) {
  middle("@(" << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  middle(')');
}

void bi::bi_ostream::visit(const FiberType* o) {
  middle(o->single << '!');
}

void bi::bi_ostream::visit(const OptionalType* o) {
  middle(o->single << '?');
}

void bi::bi_ostream::visit(const TypeIdentifier* o) {
  middle(o->name);
}

void bi::bi_ostream::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}
