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
    line("template<class Frame = EmptyFrame>");
    start(o->name->str() << "(const " << o->name->str() << "<Group>& o");
    middle(", const bool deep = true");
    middle(", const Frame& frame = EmptyFrame()");
    middle(", const char* name = nullptr");
    middle(", const MemoryGroup& group = MemoryGroup())");

    Gatherer<VarDeclaration> gatherer;
    o->braces->accept(&gatherer);

    if (o->isLess() || gatherer.size() > 0) {
      finish(" :");
      in();
      in();
      if (o->isLess()) {
        finish("base_type(o, deep, frame, name, group)");
      }
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        if (o->isLess() || iter != gatherer.begin()) {
          finish(',');
        }
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

void bi::CppCopyConstructorGenerator::initialise(const VarParameter* o) {
  start(o->name->str() << "(o." << o->name->str() << ", deep, frame, name, group)");
}
