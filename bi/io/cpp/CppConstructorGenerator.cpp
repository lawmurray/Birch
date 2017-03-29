/**
 * @file
 */
#include "bi/io/cpp/CppConstructorGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppConstructorGenerator::CppConstructorGenerator(std::ostream& base,
    const int level, const bool header) :
    indentable_ostream(base, level, header) {
  //
}

void bi::CppConstructorGenerator::visit(const BracketsType* o) {
  middle("make_frame(");
  CppBaseGenerator aux(base, level, header);
  aux << o->brackets;
  middle(")*frame.lead");
}

void bi::CppConstructorGenerator::visit(const ModelReference* o) {
  middle("frame");
}

void bi::CppConstructorGenerator::visit(const ModelParameter* o) {
  CppBaseGenerator aux(base, level, header);

  Gatherer<VarDeclaration> gatherer;
  o->braces->accept(&gatherer);

  /* two constructors are created here, one for nonempty frames, and one for
   * empty frames, this helps with debugging by ensuring that there is not a
   * catch-all constructor ("template<class Frame>") that can take any first
   * argument */
  for (int i = 0; i < 2; ++i) {
    if (header) {
      if (i == 0) {
        line("template<class Tail, class Head>");
      }
      start(o->name->str() << '(');
      if (!o->parens->isEmpty()) {
        aux << o->parens->strip();
        middle(", ");
      }
      if (i == 0) {
        middle("const NonemptyFrame<Tail,Head>& frame");
      } else {
        middle("const EmptyFrame& frame = EmptyFrame()");
      }
      middle(", const char* name = nullptr");
      middle(", const Group& group = Group()");
      middle(')');
      finish(" :");
      in();
      in();
      if (o->isLess()) {
        start("base_type(");
        ModelReference* base = dynamic_cast<ModelReference*>(o->base.get());
        assert(base);
        if (!base->parens->isEmpty()) {
          aux << base->parens->strip();
          middle(", ");
        }
        finish("frame, name, group),");
      }
      start("group(childGroup(group, name))");

      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        initialise((*iter)->param.get());
      }

      out();
      out();
      finish(" {");
      in();
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        assign((*iter)->param.get());
      }
      out();
      line("}\n");
    }
  }
}

void bi::CppConstructorGenerator::initialise(const VarParameter* o) {
  CppBaseGenerator aux(base, level, header);
  finish(',');
  start(o->name->str() << "(");
//  if (!o->parens->strip()->isEmpty() && o->type->count() == 0) {
//    aux << o->parens->strip();
//    middle(", ");
//  }
  middle(o->type);
  middle(", \"" << o->name->str() << "\"");
  middle(", childGroup");
  middle("(this->group, \"" << o->name->str() << "\")");
//  if (!o->parens->strip()->isEmpty() && o->type->count() > 0) {
//    middle(", ");
//    aux << o->value->strip();
//  }
  middle(')');
}

void bi::CppConstructorGenerator::assign(const VarParameter* o) {
  CppBaseGenerator aux(base, level, header);
  if (!o->value->isEmpty()) {
    start(o->name->str() << " = ");
    aux << o->value;
    finish(';');
  }
}
