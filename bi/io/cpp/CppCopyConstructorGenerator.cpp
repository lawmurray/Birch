/**
 * @file
 */
#include "bi/io/cpp/CppCopyConstructorGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppCopyConstructorGenerator::CppCopyConstructorGenerator(
    std::ostream& base, const int level, const bool header) :
    indentable_ostream(base, level, header) {
  //
}

void bi::CppCopyConstructorGenerator::visit(const ModelParameter* o) {
  if (header) {
    /* default copy constructor */
    line(o->name->str() << "(const " << o->name->str() << "<Group>& o) = default;\n");

    /* generic copy constructor */
    line("template<class Group1>");
    start(o->name->str() << "(const " << o->name->str() << "<Group1>& o)");

    Gatherer<VarDeclaration> gatherer;
    o->braces->accept(&gatherer);

    if (!o->base->isEmpty() || gatherer.gathered.size() > 0) {
      finish(" :");
      in();
      in();
      if (!o->base->isEmpty()) {
        finish("base_type(o)");
      }
      for (auto iter = gatherer.gathered.begin();
          iter != gatherer.gathered.end(); ++iter) {
        if (!o->base->isEmpty() || iter != gatherer.gathered.begin()) {
          finish(',');
        }
        *this << *iter;
      }
      out();
      out();
    }
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }
}

void bi::CppCopyConstructorGenerator::visit(const VarDeclaration* o) {
  start(o->param->name->str() << "(o." << o->param->name->str() << ")");
}
