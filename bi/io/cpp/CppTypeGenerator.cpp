/**
 * @file
 */
#include "bi/io/cpp/CppTypeGenerator.hpp"

#include "bi/io/cpp/CppConstructorGenerator.hpp"
#include "bi/io/cpp/CppViewConstructorGenerator.hpp"
#include "bi/io/cpp/CppCopyConstructorGenerator.hpp"
#include "bi/io/cpp/CppAssignmentGenerator.hpp"
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
      line("template<class Group = MemoryGroup>");
      start("class " << o->name);
      if (!o->base->isEmpty()) {
        middle(" : public " << o->getBase()->name << "<Group>");
      }
      finish(" {");
      line("public:");
      in();
      line("typedef Group group_type;");
      line("typedef " << o->name << "<Group> value_type;");
      if (!o->base->isEmpty()) {
        line("typedef " << o->getBase()->name << "<Group> base_type;");
      }
      line("");
    }

    /* constructor */
    CppConstructorGenerator auxConstructor(base, level, header);
    auxConstructor << o;

    /* copy constructor */
    if (header) {
      line(o->name << "(const " << o->name << "<Group>& o) = default;\n");
    }

    /* move constructor */
    if (header) {
      line(o->name << '(' << o->name << "<Group>&& o) = default;\n");
    }

    /* view constructor */
    CppViewConstructorGenerator auxViewConstructor(base, level, header);
    auxViewConstructor << o;

    /* generic copy constructor */
    CppCopyConstructorGenerator auxCopyConstructor(base, level, header);
    auxCopyConstructor << o;

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

    /* copy assignment operator */
    CppAssignmentGenerator auxAssignment(base, level, header);
    auxAssignment << o;

    /* move assignment operator */
    if (header) {
      start(o->name << "<Group>& ");
      middle("operator=(" << o->name << "<Group>&& o_)");
      finish(" = default;\n");
    }

    /* group member variable */
    if (header) {
      line("Group group;");
    }

    /* member variables and functions */
    *this << o->braces;

    /* end boilerplate */
    if (header) {
      out();
      line("};\n");
    }

    /* explicit template specialisations */
    if (!header) {
      line("template class bi::type::" << o->name << "<bi::MemoryGroup>;");
      line("//template class bi::type::" << o->name << "<bi::NetCDFGroup>;");
      line("");
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
      middle("PrimitiveValue<shared_ptr<" << o->name << "<Group>>,Group>");
    } else {
      middle(o->name << "<Group>");
    }
  }
}

void bi::CppTypeGenerator::visit(const VarDeclaration* o) {
  if (header) {
    if (o->param->type->isClass()) {
      /* use struct-of-arrays */
      start(o->param->type << ' ' << o->param->name);
    } else {
      start("PrimitiveValue<" << o->param->type << ",Group> ");
      middle(o->param->name);
    }
    finish(';');
  }
}

void bi::CppTypeGenerator::visit(const FuncParameter* o) {
  if (!o->braces->isEmpty()) {
    /* class template parameters */
    if (!header) {
      line("template<class Group>");
    }

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
      middle("bi::type::" << type->name << "<Group>::");
    }
    if ((o->isBinary() || o->isUnary()) && isTranslatable(o->name->str())) {
      middle("operator" << o->name);
    } else {
      middle(internalise(o->name->str()));
    }

    /* parameters */
    CppParameterGenerator auxParameter(base, level, header);
    auxParameter << o;

    middle(" const");
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
    /* class template parameters */
    if (!header) {
      line("template<class Group>");
    }

    /* name */
    if (!header) {
      middle("bi::type::" << type->name << "<Group>::");
    }
    middle("operator " << o->type << "() const");
    if (header) {
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
