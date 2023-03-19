/**
 * @file
 */
#include "src/generate/CppClassGenerator.hpp"

#include "src/visitor/Gatherer.hpp"
#include "src/primitive/string.hpp"

birch::CppClassGenerator::CppClassGenerator(std::ostream& base,
    const int level, const bool header, const bool includeInline,
    const bool includeLines, const Class* currentClass) :
    CppGenerator(base, level, header, includeInline, includeLines),
    currentClass(currentClass) {
  //
}

void birch::CppClassGenerator::visit(const Class* o) {
  if (!o->isAlias() && !o->braces->isEmpty()) {
    Gatherer<MemberFunction> memberFunctions;
    Gatherer<MemberVariable> memberVariables;
    Gatherer<MemberPhantom> memberPhantoms;
    o->accept(&memberFunctions);
    o->accept(&memberVariables);
    o->accept(&memberPhantoms);

    if (header) {
      genDoc(o->loc);
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("class ");
      start(o->name);
      middle('_');
      // ^ suffix class name with _, typedef actual name to Shared<Name_>
      if (o->has(FINAL)) {
        middle(" final");
      }
      middle(" : public ");
      genBase(o, false);
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
      start("MEMBIRCH_CLASS(" << o->name << "_, ");
      if (o->name->str() == "Object") {
        middle("membirch::Any");
      } else if (o->base->isEmpty()) {
        middle("Object_");
      } else {
        genBase(o, true);
      }
      finish(')');
      
      genSourceLine(o->loc);
      start("MEMBIRCH_CLASS_MEMBERS(");
      if (memberVariables.size() + memberPhantoms.size() > 0) {
        bool first = true;
        for (auto o : memberVariables) {
          if (!first) {
            middle(", ");
          }
          first = false;
          middle(o->name);
        }
        for (auto o : memberPhantoms) {
          if (!first) {
            middle(", ");
          }
          first = false;
          middle(o->name);
        }
      } else {
        middle("MEMBIRCH_NO_MEMBERS");
      }
      finish(')');
  
      /* using declarations for member functions in base classes that are
       * overridden */
      std::set<std::string> names;
      for (auto f : memberFunctions) {
        if (f->has(OVERRIDE)) {
          names.insert(f->name->str());
        }
      }

      genSourceLine(o->loc);
      line("using base_type_::operator=;");
      for (auto name : names) {
        genSourceLine(o->loc);
        line("using base_type_::" << sanitize(name) << ';');
      }
      line("");
    }

    /* constructor */
    if (!header) {
      genTemplateParams(o);
      genSourceLine(o->loc);
      start(o->name << '_');
      genTemplateArgs(o);
      middle("::");
    } else {
      genSourceLine(o->loc);
      start("");
    }
    middle(o->name << '_');
    middle('(' << o->params << ')');
    if (header) {
      finish(";\n");
    } else {
      finish(" :");
      in();
      in();
      genSourceLine(o->loc);
      start("base_type_(" << o->args << ')');
      ++inConstructor;
      bool first = false;
      for (auto o : memberVariables) {
        if (first) {
          finish(" :");
          in();
          in();
          first = false;
        } else {
          finish(',');
        }
        genSourceLine(o->loc);
        start(o->name << '(');
        genInit(o);
        middle(')');
      }
      --inConstructor;
      if (!first) {
        out();
        out();
      }
      finish(" {");
      in();
      line("//");
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

    /* factory function if default constructible */
    if (!o->has(ABSTRACT) && !o->isGeneric() && o->params->isEmpty()) {
      genSourceLine(o->loc);
      start("Object_* make_" << o->name << "_()");
      if (header) {
        line(';');
      } else {
        line(" {");
        in();
        genSourceLine(o->loc);
        line("return new " << o->name << "_();");
        genSourceLine(o->loc);
        out();
        line("}");

        if (!o->braces->isEmpty()) {
          start("static int register_factory_" << o->name);
          middle("_ = ::register_factory(");
          finish("\"" << o->name << "\", make_" << o->name << "_);");
        }
      }
      line("");
    }
  }
}

void birch::CppClassGenerator::visit(const MemberVariable* o) {
  if (header) {
    genDoc(o->loc);
    line(o->type << ' ' << o->name << ';');
  }
}

void birch::CppClassGenerator::visit(const MemberFunction* o) {
  if ((includeInline || !o->isGeneric()) && (!o->braces->isEmpty() ||
      (header && o->has(ABSTRACT)))) {
    if (header) {
      genDoc(o->loc);
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
      middle(currentClass->name << '_');
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
      *this << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void birch::CppClassGenerator::visit(const AssignmentOperator* o) {
  if (!o->braces->isEmpty()) {
    genDoc(o->loc);
    if (header) {
      genSourceLine(o->loc);
      if (!currentClass->has(FINAL)) {
        start("virtual ");
      } else {
        start("");
      }
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start("");
    }
    middle(currentClass->name << '_');
    genTemplateArgs(currentClass);
    middle("& ");
    if (!header) {
      middle(currentClass->name << '_');
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle("operator=(" << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
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
    genDoc(o->loc);
    if (header) {
      genSourceLine(o->loc);
      if (!currentClass->has(FINAL)) {
        start("virtual ");
      } else {
        start("");
      }
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start(currentClass->name << '_');
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle("operator " << o->returnType << "()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
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
    genDoc(o->loc);
    if (header) {
      genSourceLine(o->loc);
      if (!currentClass->has(FINAL)) {
        start("virtual ");
      } else {
        start("");
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
      middle(currentClass->name << '_');
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle("operator()(" << o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      ++inOperator;
      *this << o->braces->strip();
      genSourceLine(o->loc);
      --inOperator;
      out();
      finish("}\n");
    }
  }
}

void birch::CppClassGenerator::genBase(const Class* o,
    const bool includeTypename) {
  auto base = dynamic_cast<const NamedType*>(o->base);
  if (base) {
    if (includeTypename) {
      middle("typename ");
    }
    middle("membirch::unwrap_pointer<" << base->name);
    if (!base->typeArgs->isEmpty()) {
      middle('<' << base->typeArgs << '>');
    }
    middle(">::type");
  } else if (o->name->str() == "Object") {
    middle("membirch::Any");
  } else {
    middle("Object_");
  }
}
