/**
 * @file
 */
#include "bi/io/cpp/CppAssignmentGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppAssignmentGenerator::CppAssignmentGenerator(std::ostream& base,
    const int level, const bool header) :
    indentable_ostream(base, level, header) {
  //
}

void bi::CppAssignmentGenerator::visit(const ModelParameter* o) {
  if (header) {
    /* basic assignment operator */
    start(o->name->str() << "<Group>& ");
    finish(
        "operator=(const " << o->name->str() << "<Group>& o) = default;\n");

    /* generic assignment operator */
    line("template<class Group1>");
    start(o->name->str() << "<Group>&");
    middle(" operator=(const " << o->name->str() << "<Group1>& o_)");
    finish(" {");
    in();
    if (o->isLess()) {
      line("base_type::operator=(o_);");
    }
    if (o->isRandom()) {
      assign(o->missing.get());
      assign(o->pos.get());
      //assign(o->x.get());
    }

    Gatherer<VarDeclaration> gatherer;
    o->braces->accept(&gatherer);
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      assign((*iter)->param.get());
    }

    line("");
    line("return *this;");
    out();
    line("}\n");
  }
}

void bi::CppAssignmentGenerator::assign(const VarParameter* o) {
  line(o->name->str() << " = o_." << o->name->str() << ';');
}
