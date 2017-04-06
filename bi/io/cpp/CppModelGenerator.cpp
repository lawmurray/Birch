/**
 * @file
 */
#include "bi/io/cpp/CppModelGenerator.hpp"

#include "bi/io/cpp/CppConstructorGenerator.hpp"
#include "bi/io/cpp/CppViewConstructorGenerator.hpp"
#include "bi/io/cpp/CppCopyConstructorGenerator.hpp"
#include "bi/io/cpp/CppAssignmentGenerator.hpp"
#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/io/cpp/CppOutputGenerator.hpp"
#include "bi/io/cpp/CppReturnGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppModelGenerator::CppModelGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    model(nullptr) {
  //
}

void bi::CppModelGenerator::visit(const ModelParameter* o) {
  model = o;
  if (!o->isBuiltin()) {
    /* start boilerplate */
    if (header) {
      line("template<class Group = MemoryGroup>");
      start("class " << o->name);
      if (o->isLess()) {
        middle(" : public " << o->getBase()->name << "<Group>");
      }
      finish(" {");
      line("public:");
      in();
      line("typedef Group group_type;");
      line("typedef " << o->name << "<Group> value_type;");
      if (o->isLess()) {
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
      line("virtual ~" << o->name << "() {");
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
      line("template class bi::model::" << o->name << "<bi::MemoryGroup>;");
      line("//template class bi::model::" << o->name << "<bi::NetCDFGroup>;");
      line("");
    }
  }
}

void bi::CppModelGenerator::visit(const ModelReference* o) {
  if (inReturn) {
    CppBaseGenerator::visit(o);
  } else {
    if (o->isBuiltin()) {
      genBuiltin(o);
    } else if (o->polymorphic) {
      middle("bi::shared_ptr<bi::model::" << o->name << "<Group>>");
    } else {
      middle("bi::model::" << o->name << "<Group>");
    }
  }
}

void bi::CppModelGenerator::visit(const VarDeclaration* o) {
  if (header) {
    if (o->param->type->isModel() && !o->param->type->polymorphic) {
      /* use struct-of-arrays */
      start(o->param->type << ' ' << o->param->name);
    } else {
      start("PrimitiveValue<" << o->param->type << ",Group> ");
      middle(o->param->name);
    }
    finish(';');
  }
}

void bi::CppModelGenerator::visit(const FuncParameter* o) {
  if (!o->braces->isEmpty()) {
    /* class template parameters */
    if (!header) {
      line("template<class Group>");
    }

    /* return type */
    if (header && o->isVirtual()) {
      start("virtual ");
    } else {
      start("");
    }
    ++inReturn;
    middle(o->type << ' ');
    --inReturn;

    /* name */
    if (!header) {
      middle("bi::model::" << model->name << "<Group>::");
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

      /* output parameters */
      CppOutputGenerator auxOutput(base, level, header);
      auxOutput << o;

      /* body */
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces;

      /* return statement */
      if (!o->result->isEmpty()) {
        CppReturnGenerator auxReturn(base, level, header);
        auxReturn << o;
      }

      out();
      finish("}\n");
    }
  }
}
