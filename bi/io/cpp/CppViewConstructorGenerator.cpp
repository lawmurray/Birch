/**
 * @file
 */
#include "bi/io/cpp/CppViewConstructorGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppViewConstructorGenerator::CppViewConstructorGenerator(
    std::ostream& base, const int level, const bool header) :
    indentable_ostream(base, level, header) {
  //
}

void bi::CppViewConstructorGenerator::visit(const ModelParameter* o) {
  if (header) {
    line("template<class Frame, class View>");
    start("");
    middle(o->name->str());
    middle("(const " << o->name->str() << "<Group>& o");
    middle(", const Frame& frame");
    middle(", const View& view)");
    if (!o->base->isEmpty() || o->vars().size() > 0) {
      finish(" :");
      in();
      in();
      if (!o->base->isEmpty()) {
        middle("base_type(o, frame, view),");
      }
      start("group(o.group)");

      Gatherer<VarDeclaration> gatherer;
      o->braces->accept(&gatherer);
      for (auto iter = gatherer.gathered.begin();
          iter != gatherer.gathered.end(); ++iter) {
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

void bi::CppViewConstructorGenerator::visit(const VarDeclaration* o) {
  finish(',');
  start(o->param->name->str());
  middle("(o." << o->param->name->str() << ", frame, view)");
}
