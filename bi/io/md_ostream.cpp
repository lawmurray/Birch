/**
 * @file
 */
#include "bi/io/md_ostream.hpp"

bi::md_ostream::md_ostream(std::ostream& base, const std::list<File*> files) :
    bih_ostream(base),
    files(files),
    depth(1) {
  //
}

void bi::md_ostream::gen() {
  genHead("Global");
  ++depth;
  genBrief<GlobalVariable>("Variables");
  genDetailed<Function>("Functions");
  genDetailed<Fiber>("Fibers");
  genDetailed<Program>("Programs");
  genDetailed<UnaryOperator>("Unary Operators");
  genDetailed<BinaryOperator>("Binary Operators");
  --depth;
  genClasses("Classes");
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

  /* inheritance */
  if (!o->base->isEmpty()) {
    line("  * Inherits from *" << o->base << "*");
  }

  /* assignments */
  Gatherer<AssignmentOperator> gathererAssignments;
  o->accept(&gathererAssignments);
  if (gathererAssignments.size() > 0) {
    start("  * Assigns from");
    for (auto o1 : gathererAssignments) {
      middle(" *" << o1 << '*');
    }
    finish("");
  }

  /* conversions */
  Gatherer<ConversionOperator> gathererConversions;
  o->accept(&gathererConversions);
  if (gathererConversions.size() > 0) {
    start("  * Converts to");
    for (auto o1 : gathererConversions) {
      middle(" *" << o1 << '*');
    }
    finish("");
  }

  ++depth;
  genBrief<MemberVariable>("Member Variables", o);
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

void bi::md_ostream::genClasses(const std::string& name) {
  Gatherer<Class> gatherer;
  for (auto file : files) {
    file->accept(&gatherer);
  }
  if (gatherer.size() > 0) {
    genHead(name);
    ++depth;
    for (auto o : gatherer) {
      genHead(o->name->str());
      *this << o;
      line("");
      line(detailed(o->loc->doc));
      line("");
    }
    --depth;
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
