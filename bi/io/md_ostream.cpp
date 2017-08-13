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
  genHead("Global");
  ++depth;
  genOneLine<GlobalVariable>("Variables", o);
  genDetailed<Function>("Functions", o);
  genDetailed<Fiber>("Fibers", o);
  genDetailed<Program>("Programs", o);
  genDetailed<UnaryOperator>("Unary Operators", o);
  genDetailed<BinaryOperator>("Binary Operators", o);
  --depth;
  genSections<Class>("Classes", o);
}

void bi::md_ostream::visit(const Name* o) {
  middle(o->str());
}

void bi::md_ostream::visit(const Parameter* o) {
  middle(o->name << ':' << o->type);
}

void bi::md_ostream::visit(const GlobalVariable* o) {
  middle(o->name << ':' << o->type);
}

void bi::md_ostream::visit(const MemberVariable* o) {
  middle(o->name << ':' << o->type);
}

void bi::md_ostream::visit(const Function* o) {
  middle(o->name << o->params);
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::md_ostream::visit(const Fiber* o) {
  middle(o->name << o->params);
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::md_ostream::visit(const Program* o) {
  middle(o->name << o->params);
}

void bi::md_ostream::visit(const MemberFunction* o) {
  middle(o->name << o->params);
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::md_ostream::visit(const MemberFiber* o) {
  middle(o->name << o->params);
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::md_ostream::visit(const BinaryOperator* o) {
  middle(o->params->getLeft());
  middle(' ' << o->name << ' ');
  middle(o->params->getRight());
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::md_ostream::visit(const UnaryOperator* o) {
  middle(o->name << ' ' << o->params);
  if (!o->returnType->isEmpty()) {
    middle(" -> " << o->returnType);
  }
}

void bi::md_ostream::visit(const AssignmentOperator* o) {
  middle(o->single->type);
}

void bi::md_ostream::visit(const ConversionOperator* o) {
  middle(o->returnType);
}

void bi::md_ostream::visit(const Class* o) {
  /* anchor for internal links */
  line("<a name=\"" << anchor(o->name->str()) << "\"></a>\n");

  if (!o->base->isEmpty()) {
    line("  * Inherits from *" << o->base << "*\n");
  }
  line(detailed(o->loc->doc) << "\n");

  ++depth;
  genOneLine<AssignmentOperator>("Assignments", o);
  genOneLine<ConversionOperator>("Conversions", o);
  genOneLine<MemberVariable>("Member Variables", o);
  genDetailed<MemberFunction>("Member Functions", o);
  genDetailed<MemberFiber>("Member Fibers", o);
  --depth;
}

void bi::md_ostream::visit(const ListType* o) {
  middle(o->head << ", " << o->tail);
}

void bi::md_ostream::visit(const ClassType* o) {
  middle("[" << o->name << "](#" << anchor(o->name->str()) << ")");
}

void bi::md_ostream::visit(const AliasType* o) {
  middle("[" << o->name << "](#" << anchor(o->name->str()) << ")");
}

void bi::md_ostream::visit(const BasicType* o) {
  middle("[" << o->name << "](#" << anchor(o->name->str()) << ")");
}

void bi::md_ostream::visit(const IdentifierType* o) {
  middle("[" << o->name << "](#" << anchor(o->name->str()) << ")");
}

void bi::md_ostream::visit(const ArrayType* o) {
  middle(o->single << '[');
  if (o->count() > 0) {
    middle("\\_");
    for (int i = 1; i < o->count(); ++i) {
      middle(",\\_");
    }
  }
  middle(']');
}

void bi::md_ostream::visit(const ParenthesesType* o) {
  middle('(' << o->single << ')');
}

void bi::md_ostream::visit(const FunctionType* o) {
  middle("\\" << o->params);
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

void bi::md_ostream::genHead(const std::string& name) {
  finish("");
  for (int i = 0; i < depth; ++i) {
    middle('#');
  }
  middle(' ');
  finish(name);
  line("");
}
