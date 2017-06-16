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

void bi::bi_ostream::visit(const ParenthesesExpression* o) {
  *this << '(' << o->single << ')';
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

void bi::bi_ostream::visit(const BracketsExpression* o) {
  *this << o->single << '[' << o->brackets << ']';
}

void bi::bi_ostream::visit(const Parameter* o) {
  *this << o->name << ':' << o->type;
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
}

void bi::bi_ostream::visit(const GlobalVariable* o) {
  *this << o->name << ':' << o->type;
}

void bi::bi_ostream::visit(const LocalVariable* o) {
  *this << o->name << ':' << o->type;
}

void bi::bi_ostream::visit(const MemberVariable* o) {
  *this << o->name << ':' << o->type;
}

void bi::bi_ostream::visit(const Identifier<Parameter>* o) {
  *this << o->name;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
}

void bi::bi_ostream::visit(const Identifier<GlobalVariable>* o) {
  *this << o->name;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
}

void bi::bi_ostream::visit(const Identifier<LocalVariable>* o) {
  *this << o->name;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
}

void bi::bi_ostream::visit(const Identifier<MemberVariable>* o) {
  *this << o->name;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
}

void bi::bi_ostream::visit(const Identifier<Function>* o) {
  *this << o->name;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
}

void bi::bi_ostream::visit(const Identifier<Coroutine>* o) {
  *this << o->name;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
}

void bi::bi_ostream::visit(const Identifier<MemberFunction>* o) {
  *this << o->name;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
}

void bi::bi_ostream::visit(const Identifier<BinaryOperator>* o) {
  *this << o->left << ' ' << o->name << ' ' << o->right;
}

void bi::bi_ostream::visit(const Identifier<UnaryOperator>* o) {
  *this << o->name << o->single;
}

void bi::bi_ostream::visit(const List<Statement>* o) {
  *this << o->head << o->tail;
}

void bi::bi_ostream::visit(const Assignment* o) {
  *this << o->left << ' ' << o->name << ' ' << o->right;
}

void bi::bi_ostream::visit(const Function* o) {
  *this << "function " << o->name << '(' << o->parens << ')';
  if (!o->returnType->isEmpty()) {
    *this << " -> " << o->returnType;
  }
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const Coroutine* o) {
  *this << "function " << o->name << '(' << o->parens << ')';
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
  *this << "program " << o->name << '(' << o->parens << ')';
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const MemberFunction* o) {
  *this << "function " << o->name << '(' << o->parens << ')';
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
  *this << "function " << o->left << ' ' << o->name << ' ' << o->right;
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
  *this << "function " << o->name << ' ' << o->single;
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
  *this << "function " << o->name << ' ' << o->single;
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const ConversionOperator* o) {
  *this << "function -> " << o->returnType;
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const Class* o) {
  *this << indent << "class " << o->name;
  if (!o->base->isEmpty()) {
    *this << " < " << o->base;
  }
  if (!o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
}

void bi::bi_ostream::visit(const AliasType* o) {
  *this << indent << "type " << o->name << " = " << o->base;
}

void bi::bi_ostream::visit(const BasicType* o) {
  *this << indent << "type " << o->name;
}

void bi::bi_ostream::visit(const Import* o) {
  *this << indent << "import " << o->path << ";\n";
}

void bi::bi_ostream::visit(const ExpressionStatement* o) {
  *this << indent << o->single << '\n';
}

void bi::bi_ostream::visit(const If* o) {
  *this << indent << "if (" << o->cond << ')' << o->braces;
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
  *this << indent << "while (" << o->cond << ')' << o->braces << '\n';
}

void bi::bi_ostream::visit(const Return* o) {
  *this << indent << "return " << o->single << ";\n";
}

void bi::bi_ostream::visit(const Raw* o) {
  *this << indent << o->name << " {{\n";
  *this << indent << o->raw << '\n';
  *this << indent << "}}\n";
}

void bi::bi_ostream::visit(const List<Type>* o) {
  *this << o->head << ", " << o->tail;
}

void bi::bi_ostream::visit(const IdentifierType<Class>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const IdentifierType<AliasType>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const IdentifierType<BasicType>* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const BracketsType* o) {
  *this << o->single;
  if (!o->brackets->isEmpty()) {
    *this << '[' << o->brackets << ']';
  } else if (o->count() > 0) {
    *this << '[';
    for (int i = 0; i < o->count(); ++i) {
      if (i != 0) {
        *this << ',';
      }
      *this << '0';
    }
    *this << ']';
  }
}

void bi::bi_ostream::visit(const ParenthesesType* o) {
  *this << '(' << o->single << ')';
}

void bi::bi_ostream::visit(const FunctionType* o) {
  *this << "Function<(" << o->parens << ')';
  if (!o->type->isEmpty()) {
    *this << " -> " << o->type;
  }
  *this << '>';
}

void bi::bi_ostream::visit(const CoroutineType* o) {
  *this << "Coroutine<" << o->type << '>';
}
