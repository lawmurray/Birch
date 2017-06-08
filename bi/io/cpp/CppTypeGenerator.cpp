/**
 * @file
 */
#include "bi/io/cpp/CppTypeGenerator.hpp"

#include "bi/io/cpp/CppConstructorGenerator.hpp"
#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppTypeGenerator::CppTypeGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    type(nullptr) {
  //
}

void bi::CppTypeGenerator::visit(const TypeParameter* o) {
  type = o;
  if (!o->isBuiltin()) {
    /* start boilerplate */
    if (header) {
      start("class " << o->name);
      if (o->super()) {
        middle(" : public " << o->super()->name);
      }
      finish(" {");
      line("public:");
      in();
      line("typedef " << o->name << " this_type;");
      if (!o->base->isEmpty()) {
        line("typedef " << o->super()->name << " super_type;");
      }
      line("");
    }

    /* constructor */
    CppConstructorGenerator auxConstructor(base, level, header);
    auxConstructor << o;

    /* destructor */
    if (header) {
      if (o->isClass()) {
        start("virtual ");
      } else {
        start("");
      }
      finish('~' << o->name << "() {");
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
}

void bi::CppTypeGenerator::visit(const TypeReference* o) {
  if (inReturn) {
    CppBaseGenerator::visit(o);
  } else {
    if (o->isBuiltin()) {
      genBuiltin(o);
    } else if (o->isClass()) {
      middle("bi::reloc_ptr<" << o->name << '>');
    } else {
      middle(o->name);
    }
  }
}

void bi::CppTypeGenerator::visit(const VarDeclaration* o) {
  if (header) {
    line(o->param->type << ' ' << o->param->name << ';');
  }
}

void bi::CppTypeGenerator::visit(const FuncParameter* o) {
  if (!o->braces->isEmpty()) {
    /* return type */
    if (header && type->isClass()) {
      start("virtual ");
    } else {
      start("");
    }
    ++inReturn;
    middle(o->type << ' ');
    --inReturn;

    /* name */
    if (!header) {
      middle("bi::type::" << type->name << "::");
    }
    if ((o->isBinary() || o->isUnary()) && isTranslatable(o->name->str())) {
      middle("operator" << o->name);
    } else {
      middle(internalise(o->name->str()));
    }

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

void bi::CppTypeGenerator::visit(const ConversionParameter* o) {
  if (!o->braces->isEmpty()) {
    if (!header) {
      start("bi::type::" << type->name << "::");
    } else {
      start("");
    }
    middle("operator " << o->type << "()");
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
