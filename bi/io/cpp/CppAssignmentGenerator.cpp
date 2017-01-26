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
    finish("operator=(const " << o->name->str() << "<Group>& o) = default;\n");

    /* generic assignment operator */
    line("template<class Group1>");
    start(o->name->str() << "<Group>&");
    middle(" operator=(const " << o->name->str() << "<Group1>& o)");
    finish(" {");
    in();
    if (!o->base->isEmpty()) {
      line("base_type::operator=(o);");
    }

    Gatherer<VarDeclaration> gatherer;
    o->braces->accept(&gatherer);
    for (auto iter = gatherer.gathered.begin();
        iter != gatherer.gathered.end(); ++iter) {
      *this << *iter;
    }

    line("");
    line("return *this;");
    out();
    line("}\n");
  }
}

void bi::CppAssignmentGenerator::visit(const VarDeclaration* o) {
  line(o->param->name->str() << " = o." << o->param->name->str() << ';');
}
