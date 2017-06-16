/**
 * @file
 */
#include <bi/io/cpp/CppClassGenerator.hpp>
#include "bi/io/cpp/CppConstructorGenerator.hpp"
#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppClassGenerator::CppClassGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    type(nullptr) {
  //
}

void bi::CppClassGenerator::visit(const Class* o) {
  type = o;

  /* start boilerplate */
  if (header) {
    start("class " << o->name);
    if (!o->base->isEmpty()) {
      middle(" : public " << o->base);
    }
    finish(" {");
    line("public:");
    in();
    line("typedef " << o->name << " this_type;");
    if (!o->base->isEmpty()) {
      line("typedef " << o->base << " super_type;");
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

  /* member variables and functions */
  *this << o->braces;

  /* end boilerplate */
  if (header) {
    out();
    line("};\n");
  }
}

void bi::CppClassGenerator::visit(const MemberVariable* o) {
  if (header) {
    line(o->type << ' ' << o->name << ';');
  }
}

void bi::CppClassGenerator::visit(const MemberFunction* o) {
  if (!o->braces->isEmpty()) {
    /* return type */
    if (header) {
      start("virtual ");
    } else {
      start("");
    }
    ++inReturn;
    middle(o->returnType << ' ');
    --inReturn;

    /* name */
    if (!header) {
      middle("bi::type::" << type->name << "::");
    }
    middle(internalise(o->name->str()));

    /* parameters */
    CppParameterGenerator auxParameter(base, level, header);
    auxParameter << o;

    //middle(" const");
    if (header && !o->parens->hasAssignable()) {
      finish(';');
    } else {
      finish(" {");
      in();

      /* body */
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces;

      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const AssignmentOperator* o) {
  if (!o->braces->isEmpty()) {
    start("");
    if (!header) {
      middle("bi::type::");
    }
    start(type->name << "& ");
    if (!header) {
      middle("bi::type::" << type->name << "::");
    }
    middle("operator=(" << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces;
      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const ConversionOperator* o) {
  if (!o->braces->isEmpty()) {
    if (!header) {
      start("bi::type::" << type->name << "::");
    } else {
      start("");
    }
    middle("operator " << o->returnType << "()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces;
      out();
      finish("}\n");
    }
  }
}
