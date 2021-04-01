/**
 * @file
 */
#include "src/generate/CppClassGenerator.hpp"

#include "src/visitor/Gatherer.hpp"
#include "src/primitive/string.hpp"

birch::CppClassGenerator::CppClassGenerator(std::ostream& base,
    const int level, const bool header, const bool includeInline,
    const Class* currentClass) :
    CppGenerator(base, level, header, includeInline),
    currentClass(currentClass) {
  //
}

void birch::CppClassGenerator::visit(const Class* o) {
  if (!o->isAlias() && !o->braces->isEmpty()) {
    currentClass = o;
    Gatherer<MemberFunction> memberFunctions;
    Gatherer<MemberVariable> memberVariables;
    o->accept(&memberFunctions);
    o->accept(&memberVariables);

    if (header) {
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("class " << o->name);
      if (o->has(FINAL)) {
        middle(" final");
      }
      middle(" : public ");
      genBase(o);
      finish(" {");
      line("public:");
      in();

      /* generic types */
      for (auto typeParam : *o->typeParams) {
        genSourceLine(o->loc);
        line("using " << typeParam << "_ = " << typeParam << ';');
      }

      /* boilerplate */
      genSourceLine(o->loc);
      if (o->has(ABSTRACT)) {
        start("LIBBIRCH_ABSTRACT_CLASS");
      } else {
        start("LIBBIRCH_CLASS");
      }
      middle('(' << o->name << ", ");
      genBase(o);
      finish(')');
      genSourceLine(o->loc);
      if (memberVariables.size() > 0) {
        start("LIBBIRCH_MEMBERS(");
        for (auto iter = memberVariables.begin(); iter != memberVariables.end();
            ++iter) {
          if (iter != memberVariables.begin()) {
            middle(", ");
          }
          middle((*iter)->name);
        }
        finish(")");
      }

      /* using declarations for member functions in base classes that are
       * overridden */
      std::set<std::string> names;
      for (auto f : memberFunctions) {
        auto name = f->name->str();
        if (o->scope->overrides(name)) {
          names.insert(name);
        }
      }

      genSourceLine(o->loc);
      line("using base_type_::operator=;");
      for (auto name : names) {
        genSourceLine(o->loc);
        line("using base_type_::" << internalise(name) << ';');
      }
      line("");
    }

    /* constructor */
    if (!header) {
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("birch::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
    } else {
      genSourceLine(o->loc);
      start("");
    }
    middle(o->name << '(' << o->params << ')');
    if (header) {
      finish(";\n");
    } else {
      finish(" :");
      in();
      in();
      genSourceLine(o->loc);
      start("base_type_(" << o->args << ')');
      ++inConstructor;
      for (auto o : memberVariables) {
	      finish(',');
        genSourceLine(o->loc);
        start(o->name << '(');
        genInit(o);
        middle(')');
      }
      --inConstructor;
      out();
      out();
      finish(" {");
      in();
      if (o->has(ACYCLIC)) {
        line("this->libbirch::Any::acyclic_();");
      } else {
        line("//");
      }
      out();
      line("}\n");
    }

    /* member variables and functions */
    *this << o->braces->strip();

    /* end class */
    if (header) {
      out();
      line("};\n");
    }

    /* C linkage function */
    if (!o->has(ABSTRACT) && !o->isGeneric() && o->params->isEmpty()) {
      genSourceLine(o->loc);
      if (header) {
        start("extern \"C\" birch::type::" << o->name << "* ");
        finish("make_" << o->name << "_();");
      } else {
        start("birch::type::" << o->name << "* ");
        finish("birch::type::make_" << o->name << "_() {");
        in();
        genSourceLine(o->loc);
        line("return new birch::type::" << o->name << "();");
        genSourceLine(o->loc);
        out();
        line("}");
      }
      line("");
    }
  }
}

void birch::CppClassGenerator::visit(const MemberVariable* o) {
  if (header) {
    line(o->type << ' ' << o->name << ';');
  }
}

void birch::CppClassGenerator::visit(const MemberFunction* o) {
  if ((includeInline || !o->isGeneric()) && (!o->braces->isEmpty() ||
      (header && o->has(ABSTRACT)))) {
    if (header) {
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("");
      if (o->typeParams->isEmpty() && !currentClass->has(FINAL)) {
        middle("virtual ");
      }
    } else {
      genTemplateParams(currentClass);
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("");
    }
    middle(o->returnType << ' ');
    if (!header) {
      middle("birch::type::" << currentClass->name);
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle(o->name << '(' << o->params << ')');
    if (header) {
      if (o->has(FINAL) && !o->isGeneric()) {
        middle(" final");
      } else if (o->has(OVERRIDE)) {
        middle(" override");
      } else if (o->has(ABSTRACT)) {
        middle(" = 0");
      }
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      *this << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void birch::CppClassGenerator::visit(const AssignmentOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      start("");
      if (!currentClass->has(FINAL)) {
        genSourceLine(o->loc);
        middle("virtual ");
      }
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start("birch::type::");
    }
    middle(currentClass->name);
    genTemplateArgs(currentClass);
    middle("& ");
    if (!header) {
      middle("birch::type::" << currentClass->name);
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle("operator=(" << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction("<assignment>", o->loc);
      ++inOperator;
      *this << o->braces->strip();
      genSourceLine(o->loc);
      line("return *this;");
      --inOperator;
      out();
      finish("}\n");
    }
  }
}

void birch::CppClassGenerator::visit(const ConversionOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      genSourceLine(o->loc);
      start("");
      if (!currentClass->has(FINAL)) {
        middle("virtual ");
      }
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start("birch::type::" << currentClass->name);
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle("operator " << o->returnType << "()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction("<conversion>", o->loc);
      ++inOperator;
      *this << o->braces->strip();
      --inOperator;
      out();
      finish("}\n");
    }
  }
}

void birch::CppClassGenerator::visit(const SliceOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      genSourceLine(o->loc);
      start("");
      if (!currentClass->has(FINAL)) {
        middle("virtual ");
      }
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start("");
    }
    middle(o->returnType);
    if (!o->returnType->isEmpty()) {
      middle("& ");
    } else {
      middle(' ');
    }
    if (!header) {
      middle("birch::type::" << currentClass->name);
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle("operator()(" << o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction("<slice>", o->loc);
      ++inOperator;
      *this << o->braces->strip();
      genSourceLine(o->loc);
      --inOperator;
      out();
      finish("}\n");
    }
  }
}

void birch::CppClassGenerator::genBase(const Class* o) {
  auto base = dynamic_cast<const NamedType*>(o->base);
  if (base) {
    middle(base->name);
    if (!base->typeArgs->isEmpty()) {
      middle('<' << base->typeArgs << '>');
    }
  } else if (o->name->str() == "Object") {
    middle("libbirch::Any");
  } else {
    middle("Object");
  }
}