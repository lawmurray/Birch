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

void bi::bi_ostream::visit(const BooleanLiteral* o) {
  *this << o->str;
}

void bi::bi_ostream::visit(const IntegerLiteral* o) {
  *this << o->str;
}

void bi::bi_ostream::visit(const RealLiteral* o) {
  *this << o->str;
}

void bi::bi_ostream::visit(const StringLiteral* o) {
  *this << o->str;
}

void bi::bi_ostream::visit(const Name* o) {
  *this << o->str();
}

void bi::bi_ostream::visit(const Path* o) {
  *this << o->str();
}

void bi::bi_ostream::visit(const ExpressionList* o) {
  *this << o->head << ", " << o->tail;
}

void bi::bi_ostream::visit(const StatementList* o) {
  *this << o->head << o->tail;
}

void bi::bi_ostream::visit(const TypeList* o) {
  *this << o->head << ", " << o->tail;
}

void bi::bi_ostream::visit(const ParenthesesExpression* o) {
  *this << '(' << o->single << ')';
}

void bi::bi_ostream::visit(const BracesExpression* o) {
  *this << " {\n";
  in();
  *this << o->single;
  out();
  *this << indent << '}';
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

void bi::bi_ostream::visit(const VarReference* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const FuncReference* o) {
  if (o->isBinary()) {
    *this << o->getLeft() << ' ' << o->name << ' ' << o->getRight();
  } else if (o->isUnary()) {
    *this << o->name << o->getRight();
  } else {
    *this << o->name << '(' << o->parens << ')';
  }
}

void bi::bi_ostream::visit(const TypeReference* o) {
  *this << o->name;
}

void bi::bi_ostream::visit(const ProgReference* o) {
  *this << o->name << '(' << o->parens << ')';
}

void bi::bi_ostream::visit(const VarParameter* o) {
  *this << o->name << ':' << o->type;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
}

void bi::bi_ostream::visit(const FuncParameter* o) {
  if (o->isBinary()) {
    *this << '(' << o->getLeft() << ' ' << o->name << ' ' << o->getRight() << ')';
  } else if (o->isUnary()) {
    *this << '(' << o->name << o->getRight() << ')';
  } else {
    *this << o->name << '(' << o->parens << ')';
  }
  if (!o->type->isEmpty()) {
    *this << " -> " << o->type;
  }
}

void bi::bi_ostream::visit(const ConversionParameter* o) {
  *this << " -> " << o->type;
}

void bi::bi_ostream::visit(const TypeParameter* o) {
  *this << o->name;
  if (!o->parens->isEmpty()) {
    *this << '(' << o->parens << ')';
  }
  if (!o->isAlias()) {
    *this << " = " << o->base;
  } else if (!o->base->isEmpty()) {
    *this << " < " << o->base;
  }
}

void bi::bi_ostream::visit(const ProgParameter* o) {
  *this << o->name << '(' << o->parens << ')';
  if (!header && !o->braces->isEmpty()) {
    *this << o->braces;
  } else {
    *this << ';';
  }
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

void bi::bi_ostream::visit(const File* o) {
  *this << o->root;
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

void bi::bi_ostream::visit(const VarDeclaration* o) {
  *this << indent << o->param << ";\n";
}

void bi::bi_ostream::visit(const FuncDeclaration* o) {
  *this << indent << "function " << o->param;
  if (!header && !o->param->braces->isEmpty()) {
    *this << o->param->braces;
  } else {
    *this << ';';
  }
  *this << "\n\n";
}

void bi::bi_ostream::visit(const ConversionDeclaration* o) {
  *this << indent << "function " << o->param;
  if (!header && !o->param->braces->isEmpty()) {
    *this << o->param->braces;
  } else {
    *this << ';';
  }
  *this << "\n\n";
}

void bi::bi_ostream::visit(const TypeDeclaration* o) {
  *this << indent;
  if (o->param->isStruct()) {
    *this << "struct";
  } else if (o->param->isClass()) {
    *this << "class";
  } else {
    *this << "type";
  }
  *this << ' ' << o->param;
  if (!o->param->braces->isEmpty()) {
    *this << o->param->braces;
  } else {
    *this << ';';
  }
  *this << "\n\n";
}

void bi::bi_ostream::visit(const ProgDeclaration* o) {
  *this << indent << "program " << o->param;
  if (!header && !o->param->braces->isEmpty()) {
    *this << o->param->braces;
  } else {
    *this << ';';
  }
  *this << "\n\n";
}
