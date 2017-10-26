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
  genHead("Summary");
  ++depth;
  genOneLine<GlobalVariable>("Variable", o, true);
  genBrief<Function>("Function", o, true);
  genBrief<Fiber>("Fiber", o, true);
  genBrief<Program>("Program", o, true);
  genBrief<UnaryOperator>("Unary Operator", o, true);
  genBrief<BinaryOperator>("Binary Operator", o, true);
  genBrief<Basic>("Basic Type", o, true);
  genBrief<Alias>("Alias Type", o, true);
  genBrief<Class>("Class Type", o, true);
  --depth;

  genDetailed<Function>("Function Details", o, true);
  genDetailed<Fiber>("Fiber Details", o, true);
  genDetailed<Program>("Program Details", o, true);
  genDetailed<UnaryOperator>("Unary Operator Details", o, true);
  genDetailed<BinaryOperator>("Binary Operator Details", o, true);
  genDetailed<Basic>("Basic Type Details", o, true);
  genDetailed<Alias>("Alias Type Details", o, true);
  genSections<Class>("Class Type Details", o, true);
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
  auto iter = o->params->begin();
  middle(*(iter++) << ' ' << o->name << ' ' << *(iter++));
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
  line("<a name=\"" << anchor(o->name->str(), o->number) << "\"></a>\n");

  if (!o->base->isEmpty()) {
    line("  * Inherits from *" << o->base << "*\n");
  }
  line(detailed(o->loc->doc) << "\n");

  ++depth;
  genOneLine<AssignmentOperator>("Assignment", o, false);
  genOneLine<ConversionOperator>("Conversion", o, false);
  genOneLine<MemberVariable>("Member Variable", o, false);
  genBrief<MemberFunction>("Member Function", o, false);
  genBrief<MemberFiber>("Member Fiber", o, false);
  genDetailed<MemberFunction>("Member Function Details", o, true);
  genDetailed<MemberFiber>("Member Fiber Details", o, true);
  --depth;
}

void bi::md_ostream::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

void bi::md_ostream::visit(const ClassType* o) {
  middle("[" << o->name << "](#" << anchor(o->name->str(), o->target->number) << ")");
}

void bi::md_ostream::visit(const AliasType* o) {
  middle("[" << o->name << "](#" << anchor(o->name->str(), o->target->number) << ")");
}

void bi::md_ostream::visit(const BasicType* o) {
  middle("[" << o->name << "](#" << anchor(o->name->str(), o->target->number) << ")");
}

void bi::md_ostream::visit(const ArrayType* o) {
  middle(o->single << "\\[");
  if (o->count() > 0) {
    middle("\\_");
    for (int i = 1; i < o->count(); ++i) {
      middle(",\\_");
    }
  }
  middle("\\]");
}

void bi::md_ostream::visit(const TupleType* o) {
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
