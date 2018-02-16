/**
 * @file
 */
#include "bi/io/md_ostream.hpp"

bi::md_ostream::md_ostream(std::ostream& base) :
    bih_ostream(base),
    depth(1) {
  //
}

void bi::md_ostream::visit(const Package* o) {
  genOneLine<Basic>("Types", o, true, false);
  genOneLine<GlobalVariable>("Variables", o, true, true);
  genDetailed<Program>("Programs", o, true, true);
  genDetailed<Function>("Functions", o, true, true);
  genDetailed<Fiber>("Fibers", o, true, true);
  genDetailed<UnaryOperator>("Unary Operators", o, true, false);
  genDetailed<BinaryOperator>("Binary Operators", o, true, false);
  genSections<Class>("Classes", o, true, false);
}

void bi::md_ostream::visit(const Name* o) {
  middle(o->str());
}

void bi::md_ostream::visit(const Parameter* o) {
  middle(o->name << ':' << o->type);
  if (!o->value->isEmpty()) {
    *this << " <- " << o->value;
  }
}

void bi::md_ostream::visit(const GlobalVariable* o) {
  middle(o->name << ':' << o->type);
}

void bi::md_ostream::visit(const MemberVariable* o) {
  middle(o->name << ':' << o->type);
}

void bi::md_ostream::visit(const Function* o) {
  start("!!! note \"function " << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const Fiber* o) {
  start("!!! note \"fiber " << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const Program* o) {
  start("!!! note \"program " << o->name << '(' << o->params << ")\"");
}

void bi::md_ostream::visit(const MemberFunction* o) {
  start("!!! note \"function");
  middle(' ' << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const MemberFiber* o) {
  start("!!! note \"fiber");
  middle(' ' << o->name << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const BinaryOperator* o) {
  start("!!! note \"operator (");
  middle(o->params->getLeft() << ' ' << o->name << ' ');
  middle(o->params->getRight() << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const UnaryOperator* o) {
  start("!!! note \"operator (");
  middle(o->name << ' ' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
  finish("\"");
}

void bi::md_ostream::visit(const AssignmentOperator* o) {
  middle(o->single->type);
}

void bi::md_ostream::visit(const ConversionOperator* o) {
  middle(o->returnType);
}

void bi::md_ostream::visit(const Basic* o) {
  middle(o->name);
  if (!o->base->isEmpty()) {
    if (o->isAlias()) {
      middle(" = ");
    } else {
      middle(" < ");
    }
    middle(o->base);
  }
}

void bi::md_ostream::visit(const Class* o) {
  /* anchor for internal links */
  line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");

  start("!!! note \"class " << o->name);
  if (o->isGeneric()) {
    middle("&lt;" << o->typeParams << "&gt;");
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
  finish("\"\n");
  line(quote(detailed(o->loc->doc), "    ") << "\n");

  ++depth;
  genOneLine<AssignmentOperator>("Assignments", o, false, false);
  genOneLine<ConversionOperator>("Conversions", o, false, false);
  genOneLine<MemberVariable>("Member Variables", o, false, true);
  genBrief<MemberFunction>("Member Functions", o, false, true);
  genBrief<MemberFiber>("Member Fibers", o, false, true);

  genDetailed<MemberFunction>("Member Function Details", o, true, true);
  genDetailed<MemberFiber>("Member Fiber Details", o, true, true);
  --depth;
}

void bi::md_ostream::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

void bi::md_ostream::visit(const ClassType* o) {
  middle('[' << o->name << "](/classes/" << o->name->str() << ".md)");
  if (!o->typeArgs->isEmpty()) {
    middle("&lt;" << o->typeArgs << "&gt;");
  }
}

void bi::md_ostream::visit(const BasicType* o) {
  middle('[' << o->name << "](/types.md#" << o->name->str() << ')');
}

void bi::md_ostream::visit(const ArrayType* o) {
  middle(o->single << '[');
  for (int i = 0; i < o->depth(); ++i) {
    if (i > 0) {
      middle(',');
    }
    middle("\\_");
  }
  middle(']');
}

void bi::md_ostream::visit(const TupleType* o) {
  middle('(' << o->single << ')');
}

void bi::md_ostream::visit(const SequenceType* o) {
  middle('[' << o->single << ']');
}

void bi::md_ostream::visit(const FunctionType* o) {
  middle('@' << '(' << o->params << ')');
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::md_ostream::visit(const FiberType* o) {
  middle(o->single << '!');
}

void bi::md_ostream::visit(const OptionalType* o) {
  middle(o->single << '?');
}

void bi::md_ostream::visit(const PointerType* o) {
  middle(o->single);
  if (o->read) {
    middle('\'');
  }
  if (o->weak) {
    middle('&');
  }
}

void bi::md_ostream::genHead(const std::string& name) {
  finish("");
  for (int i = 0; i < depth; ++i) {
    middle('#');
  }
  middle(' ');
  finish(name);
  line("");
}
