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
    Gatherer<VarDeclaration> gatherer;
    o->braces->accept(&gatherer);
    if (o->isLess() || o->isRandom() || gatherer.size() > 0) {
      finish(" :");
      in();
      in();
      if (o->isLess()) {
        middle("base_type(o, frame, view),");
      }
      start("group(o.group)");
      if (o->isRandom()) {
        initialise(o->missing.get());
        initialise(o->pos.get());
        initialise(o->x.get());
      }
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        initialise((*iter)->param.get());
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

void bi::CppViewConstructorGenerator::initialise(const VarParameter* o) {
  finish(',');
  start(o->name->str());
  middle("(o." << o->name->str() << ", frame, view)");
}
