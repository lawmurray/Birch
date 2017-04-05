/**
 * @file
 */
#include "bi/io/cpp/CppConstructorGenerator.hpp"

bi::CppConstructorGenerator::CppConstructorGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppConstructorGenerator::visit(const ModelParameter* o) {
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
        middle(o->parens << ", ");
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
          middle(base->parens << ", ");
        }
        finish("frame, name, group),");
      }
      start("group(childGroup(group, name))");

      inInitial = true;
      *this << o->braces;

      out();
      out();
      finish(" {");
      in();
      inInitial = false;
      *this << o->braces;
      out();
      line("}\n");
    }
  }
}

void bi::CppConstructorGenerator::visit(const VarParameter* o) {
  if (inInitial) {
    finish(',');
    start(o->name << "(");
  //  if (!o->parens()->isEmpty() && o->type->count() == 0) {
  //    aux << o->parens->strip();
  //    middle(", ");
  //  }
    if (o->type->isArray()) {
      const BracketsType* type = dynamic_cast<const BracketsType*>(o->type.get());
      assert(type);
      middle("make_frame(" << type->brackets << ")*frame.lead");
    } else {
      middle("frame");
    }
    middle(", \"" << o->name << "\"");
    middle(", childGroup");
    middle("(this->group, \"" << o->name << "\")");
  //  if (!o->parens->isEmpty() && o->type->count() > 0) {
  //    middle(", "  << o->parens);
  //  }
    middle(')');
  } else {
    if (!o->value->isEmpty()) {
      line("this->group.fill(" << o->name << ", " << o->value << "(), frame);");
    } else if (o->type->isLambda() || o->type->isDelay() || o->type->polymorphic) {
      line("this->group.fill(" << o->name << ", " << o->type << "(), frame);");
    }
  }
}

void bi::CppConstructorGenerator::visit(const VarDeclaration* o) {
  *this << o->param;
}

void bi::CppConstructorGenerator::visit(const FuncDeclaration* o) {
  //
}
