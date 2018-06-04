/**
 * @file
 */
#include "bi/io/cpp/CppClassGenerator.hpp"

#include "bi/io/cpp/CppConstructorGenerator.hpp"
#include "bi/io/cpp/CppMemberFiberGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppClassGenerator::CppClassGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    type(nullptr) {
  //
}

void bi::CppClassGenerator::visit(const Class* o) {
  if (o->isAlias() || !o->braces->isEmpty()) {
    type = o;

    /* start boilerplate */
    if (header) {
      if (!o->isAlias()) {
        genTemplateParams(o);
        start("class " << o->name);
        if (o->isInstantiation) {
          genTemplateArgs(o);
        }
        if (!o->base->isEmpty()) {
          middle(" : public ");
          middle(o->base);
        }
        finish(" {");
        line("public:");
        in();
        if (!o->isGeneric() || o->isInstantiation) {
          start("using this_type = " << o->name);
          genTemplateArgs(o);
          finish(';');
          if (!o->base->isEmpty()) {
            line("using super_type = " << o->base << ';');
          }
        }
        line("");
      }
    }

    if (!o->isAlias() && (!o->isGeneric() || o->isInstantiation)) {
      /* constructor */
      CppConstructorGenerator auxConstructor(base, level, header);
      auxConstructor << o;

      /* destructor */
      if (header) {
        line("virtual ~" << o->name << "() {");
        in();
        line("//");
        out();
        line("}\n");
      }

      /* ensure super type assignments are visible */
      if (header) {
        line("using super_type::operator=;\n");
      }

      /* self-reference functions */
      if (header) {
        line("this_type* self() {");
        in();
        line("return this;");
        out();
        line("}\n");

        line("super_type* super() {");
        in();
        line("return this;");
        out();
        line("}\n");

        line("SharedPointer<this_type> shared_self() {");
        in();
        line("return shared_from_this<this_type>();");
        out();
        line("}\n");

        line("SharedPointer<super_type> shared_super() {");
        in();
        line("return shared_from_this<super_type>();");
        out();
        line("}\n");
      }

      /* clone function */
      if (!header) {
        start("");
      } else {
        start("virtual ");
      }
      middle("std::shared_ptr<bi::Any> ");
      if (!header) {
        middle("bi::type::" << o->name);
        genTemplateArgs(o);
        middle("::");
      }
      middle("clone() const");
      if (header) {
        finish(";\n");
      } else {
        finish(" {");
        in();
        line("return std::make_shared<this_type>(*this);");
        out();
        line("}\n");
      }

      /* member variables and functions */
      *this << o->braces->strip();
    }

    /* end class */
    if (!o->isAlias() && header) {
      out();
      line("};\n");
    }

    /* C linkage function */
    if (!o->isGeneric() && o->params->isEmpty()) {
      if (header) {
        line("extern \"C\" " << o->name << "* make_" << o->name << "();");
      } else {
        line(
            "bi::type::" << o->name << "* bi::type::make_" << o->name << "() {");
        in();
        line("return new bi::type::" << o->name << "();");
        out();
        line("}");
      }
      line("");
    }
  }
}

void bi::CppClassGenerator::visit(const MemberVariable* o) {
  if (header) {
    line(o->type << ' ' << o->name << ';');
  }
}

void bi::CppClassGenerator::visit(const MemberFunction* o) {
  if (header) {
    start("virtual ");
  } else {
    start("");
  }
  middle(o->returnType << ' ');
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::");
  }
  middle(internalise(o->name->str()) << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    line("Enter enter(getWorld());");

    /* body */
    CppBaseGenerator auxBase(base, level, header);
    auxBase << o->braces->strip();

    out();
    finish("}\n");
  }
}

void bi::CppClassGenerator::visit(const MemberFiber* o) {
  CppMemberFiberGenerator auxMemberFiber(type, base, level, header);
  auxMemberFiber << o;
}

void bi::CppClassGenerator::visit(const AssignmentOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      start("virtual ");
    } else {
      start("bi::type::");
    }
    middle(type->name);
    genTemplateArgs(type);
    middle("& ");
    if (!header) {
      middle("bi::type::" << type->name);
      genTemplateArgs(type);
      middle("::");
    }
    middle("operator=(" << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction("(assignment)", o->loc);
      line("Enter enter(getWorld());");
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      line("return *this;");
      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const ConversionOperator* o) {
  if (!o->braces->isEmpty()) {
    if (!header) {
      start("bi::type::" << type->name);
      genTemplateArgs(type);
      middle("::");
    } else {
      start("virtual ");
    }
    middle("operator " << o->returnType << "()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction("(conversion)", o->loc);
      line("Enter enter(getWorld());");
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}
