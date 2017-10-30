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
  if (!o->braces->isEmpty()) {
    type = o;

    /* start boilerplate */
    if (header) {
      genTemplateParams(o);
      start("class " << o->name << " : public ");
      if (!o->base->isEmpty()) {
        auto type = dynamic_cast<const ClassType*>(o->base);
        assert(type);
        middle(type->name);
        if (!type->typeArgs->isEmpty()) {
          middle('<' << type->typeArgs << '>');
        }
      } else {
        middle("Object_");
      }
      finish(" {");
      line("public:");
      in();
      start("typedef " << o->name);
      genTemplateArgs(o);
      finish(" this_type;");
      if (o->base->isEmpty()) {
        line("typedef Object_ super_type;");
      } else {
        auto type = dynamic_cast<const ClassType*>(o->base);
        assert(type);
        start("typedef " << type->name);
        if (!type->typeArgs->isEmpty()) {
          middle('<' << type->typeArgs << '>');
        }
        finish(" super_type;");
      }
      line("");
    }

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

    /* clone function */
    if (!header) {
      genTemplateParams(o);
      start("bi::");
    } else {
      start("virtual ");
    }
    middle(o->name);
    genTemplateArgs(o);
    middle("* ");
    if (!header) {
      middle("bi::" << o->name);
      genTemplateArgs(o);
      middle("::");
    }
    middle("clone()");
    if (header) {
      finish(";\n");
    } else {
      finish(" {");
      in();
      line("return copy_object(this);");
      out();
      line("}\n");
    }

    /* member parameters */
    for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
      *this << *iter;
    }

    /* member variables and functions */
    *this << o->braces->strip();

    /* end boilerplate */
    if (header) {
      out();
      line("};\n");
    }
  }
}

void bi::CppClassGenerator::visit(const MemberParameter* o) {
  if (header) {
    line(o->type << ' ' << o->name << ';');
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
    genTemplateParams(type);
    start("");
  }
  middle(o->returnType << ' ');
  if (!header) {
    middle("bi::" << type->name);
    genTemplateArgs(type);
    middle("::");
  }
  middle(internalise(o->name->str()) << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();

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
      genTemplateParams(type);
      start("bi::");
    }
    middle(type->name);
    genTemplateArgs(type);
    middle("& ");
    if (!header) {
      middle("bi::" << type->name);
      genTemplateArgs(type);
      middle("::");
    }
    middle("operator=(" << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
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
      genTemplateParams(type);
      start("bi::" << type->name);
      genTemplateArgs(type);
      middle("::");
    } else {
      /* user-defined conversions should be marked explicit to work properly
       * with the Pointer class in the compiler library; see also
       * has_conversion */
      start("virtual explicit ");
    }
    middle("operator " << o->returnType << "()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}
