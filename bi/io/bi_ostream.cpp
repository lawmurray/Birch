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

void bi::bi_ostream::visit(const File* o) {
  *this << o->root;
}

void bi::bi_ostream::visit(const Name* o) {
  *this << o->str();
}

void bi::bi_ostream::visit(const Path* o) {
  *this << o->str();
}

void bi::bi_ostream::visit(const List<Expression>* o) {
  *this << o->head << ", " << o->tail;
}

void bi::bi_ostream::visit(const Literal<bool>* o) {
  *this << o->str;
}

void bi::bi_ostream::visit(const Literal<int64_t>* o) {
  *this << o->str;
}

void bi::bi_ostream::visit(const Literal<double>* o) {
  *this << o->str;
}

void bi::bi_ostream::visit(const Literal<const char*>* o) {
  *this << o->str;
}

void bi::bi_ostream::visit(const Parentheses* o) {
  *this << '(' << o->single << ')';
}

void bi::bi_ostream::visit(const Brackets* o) {
  *this << '[' << o->single << ']';
}

void bi::bi_ostream::visit(const Call* o) {
  *this << o->single << o->args;
}

void bi::bi_ostream::visit(const BinaryCall* o) {
  *this << o->args->getLeft() << ' ' << o->single << ' ' << o->args->getRight();
}

void bi::bi_ostream::visit(const UnaryCall* o) {
  *this << o->single << o->args;
}

void bi::bi_ostream::visit(const Slice* o) {
  *this << o->single << o->brackets;
}

void bi::bi_ostream::visit(const Query* o) {
  *this << o->single << '?';
}

void bi::bi_ostream::visit(const Get* o) {
  *this << o->single << '!';
}

void bi::bi_ostream::visit(const Index* o) {
  *this << o->single;
}

void bi::bi_ostream::visit(const Range* o) {
  *this << o->left << ".." << o->right;
}

void bi::bi_ostream::visit(const Member* o) {
  *this << o->left << '.' << o->right;
}

void bi::bi_ostream::visit(const Super* o) {
  *this << "super";
}

void bi::bi_ostream::visit(const This* o) {
  *this << "this";
}

void bi::bi_ostream::visit(const Nil* o) {
  *this << "nil";
}

void bi::bi_ostream::visit(const LocalVariable* o) {
  *this << o->name << ':' << o->type;
  if (!o->parens->isEmpty()) {
    *this << o->value;
  }
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
}

void bi::bi_ostream::visit(const Parameter* o) {
  *this << o->name << ':' << o->type;
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
}

void bi::bi_ostream::visit(const MemberParameter* o) {
  *this << o->name << ':' << o->type;
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
}

void bi::bi_ostream::visit(const GlobalVariable* o) {
  *this << indent << o->name << ':' << o->type;
  if (!o->parens->isEmpty()) {
    *this << o->value;
  }
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
  *this << ';';
}

void bi::bi_ostream::visit(const MemberVariable* o) {
  *this << indent << o->name << ':' << o->type;
  if (!o->parens->isEmpty()) {
    *this << o->value;
  }
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
  *this << ';';
}

void bi::bi_ostream::visit(const Identifier<Parameter>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const Identifier<GlobalVariable>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const Identifier<LocalVariable>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const Identifier<MemberVariable>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const OverloadedIdentifier<Function>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const OverloadedIdentifier<Fiber>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const OverloadedIdentifier<MemberFunction>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const OverloadedIdentifier<MemberFiber>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const OverloadedIdentifier<BinaryOperator>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const OverloadedIdentifier<UnaryOperator>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const Identifier<Unknown>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const List<Statement>* o) {
  *this << o->head << o->tail;
}

void bi::bi_ostream::visit(const Assignment* o) {
  *this << o->left << ' ' << o->name << ' ' << o->right;
}

void bi::bi_ostream::visit(const Function* o) {
  *this << "function " << o->name << o->params;
  if (!o->returnType->isEmpty()) {
    *this << " -> " << o->returnType;
  }
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const Fiber* o) {
  *this << "fiber " << o->name << o->params;
  if (!o->returnType->isEmpty()) {
    *this << " -> " << o->returnType;
  }
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const Program* o) {
  *this << "program " << o->name << o->params;
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const MemberFunction* o) {
  *this << "function " << o->name << o->params;
  if (!o->returnType->isEmpty()) {
    *this << " -> " << o->returnType;
  }
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const MemberFiber* o) {
  *this << "fiber " << o->name << o->params;
  if (!o->returnType->isEmpty()) {
    *this << " -> " << o->returnType;
  }
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const BinaryOperator* o) {
  *this << "operator ";
  *this << o->params->getLeft();
  *this << ' ' << o->name << ' ';
  *this << o->params->getRight();
  if (!o->returnType->isEmpty()) {
    *this << " -> " << o->returnType;
  }
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const UnaryOperator* o) {
  *this << "operator " << o->name << ' ' << o->params;
  if (!o->returnType->isEmpty()) {
    *this << " -> " << o->returnType;
  }
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const AssignmentOperator* o) {
  *this << "operator " << o->name << ' ' << o->single;
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const ConversionOperator* o) {
  *this << "operator -> " << o->returnType;
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const Class* o) {
  *this << indent << "class " << o->name;
  if (!o->parens->isEmpty()) {
    *this << o->parens;
  }
  if (!o->base->isEmpty()) {
    *this << " < " << o->base;
  }
  if (!o->baseParens->isEmpty()) {
    *this << o->baseParens;
  }
  if (!o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const Alias* o) {
  *this << indent << "type " << o->name << " = " << o->base;
}

void bi::bi_ostream::visit(const Basic* o) {
  *this << indent << "type " << o->name;
}

void bi::bi_ostream::visit(const Import* o) {
  *this << indent << "import " << o->path << ";\n";
}

void bi::bi_ostream::visit(const ExpressionStatement* o) {
  *this << indent << o->single << '\n';
}

void bi::bi_ostream::visit(const If* o) {
  *this << indent << "if " << o->cond << o->braces;
  if (!o->falseBraces->isEmpty()) {
    *this << " else" << o->falseBraces;
  }
  *this << '\n';
}

void bi::bi_ostream::visit(const For* o) {
  *this << indent << "for (";
  *this << o->index << " in " << o->from << ".." << o->to;
  *this << ')' << o->braces << '\n';
}

void bi::bi_ostream::visit(const While* o) {
  *this << indent << "while " << o->cond << o->braces << '\n';
}

void bi::bi_ostream::visit(const Assert* o) {
  *this << indent << "assert " << o->cond << ";\n";
}

void bi::bi_ostream::visit(const Return* o) {
  *this << indent << "return " << o->single << ";\n";
}

void bi::bi_ostream::visit(const Yield* o) {
  *this << indent << "yield " << o->single << ";\n";
}

void bi::bi_ostream::visit(const Raw* o) {
  *this << indent << o->name << " {{\n";
  *this << indent << o->raw << '\n';
  *this << indent << "}}\n";
}

void bi::bi_ostream::visit(const ListType* o) {
  *this << o->head << ", " << o->tail;
}

void bi::bi_ostream::visit(const ClassType* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const AliasType* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const BasicType* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const BinaryType* o) {
  *this << '(' << o->left << ", " << o->right << ')';
}

void bi::bi_ostream::visit(const IdentifierType* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const ArrayType* o) {
  *this << o->single;
  if (!o->brackets->isEmpty()) {
    *this << '[' << o->brackets << ']';
  } else if (o->count() > 0) {
    *this << '[';
    for (int i = 0; i < o->count(); ++i) {
      if (i != 0) {
        *this << ',';
      }
      *this << '_';
    }
    *this << ']';
  }
}

void bi::bi_ostream::visit(const ParenthesesType* o) {
  *this << '(' << o->single << ')';
}

void bi::bi_ostream::visit(const FunctionType* o) {
  *this << "Function<" << o->params;
  if (!o->returnType->isEmpty()) {
    *this << " -> " << o->returnType;
  }
  *this << '>';
}

void bi::bi_ostream::visit(const FiberType* o) {
  *this << o->single << '!';
}

void bi::bi_ostream::visit(const OptionalType* o) {
  *this << o->single << '?';
}
