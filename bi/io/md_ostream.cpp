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
  genHead("Programs");
  genSection<Program>();
  genHead("Variables");
  genSection<GlobalVariable>();
  genHead("Functions");
  genSection<Function>();
  genHead("Fibers");
  genSection<Fiber>();
  genHead("Operators");
  genSection<UnaryOperator>();
  genSection<BinaryOperator>();
  --depth;
  genHead("Classes");
  genSection<Class>();
}

void bi::md_ostream::visit(const Class* o) {
  out(); out(); // cancel bih_ostream indenting
  ++depth;
  genHead("Member Variables");
  genClassSection<MemberVariable>(o);
  genHead("Member Functions");
  genClassSection<MemberFunction>(o);
  genHead("Member Fibers");
  genClassSection<MemberFiber>(o);
  genHead("Assignments");
  genClassSection<AssignmentOperator>(o);
  genHead("Conversions");
  genClassSection<ConversionOperator>(o);
  --depth;
  in(); in();
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
