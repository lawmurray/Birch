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
    o->accept(&memberFunctions);
    o->accept(&memberVariables);

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
      start("LIBBIRCH_CLASS(" << o->name << "_, ");
      if (o->name->str() == "Object") {
        middle("libbirch::Any");
      } else if (o->base->isEmpty()) {
        middle("Object_");
      } else {
        genBase(o, true);
      }
      finish(')');
      
      genSourceLine(o->loc);
      start("LIBBIRCH_CLASS_MEMBERS(");
      if (memberVariables.size() > 0) {
        for (auto iter = memberVariables.begin(); iter != memberVariables.end();
            ++iter) {
          if (iter != memberVariables.begin()) {
            middle(", ");
          }
          middle((*iter)->name);
        }
      } else {
        middle("LIBBIRCH_NO_MEMBERS");
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

    /* C linkage function */
    if (!o->has(ABSTRACT) && !o->isGeneric() && o->params->isEmpty()) {
      genSourceLine(o->loc);
      if (header) {
        line("extern \"C\" " << o->name << "_* make_" << o->name << "_();");
      } else {
        line(o->name << "_* make_" << o->name << "_() {");
        in();
        genSourceLine(o->loc);
        line("return new " << o->name << "_();");
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
    middle("libbirch::unwrap_pointer<" << base->name);
    if (!base->typeArgs->isEmpty()) {
      middle('<' << base->typeArgs << '>');
    }
    middle(">::type");
  } else if (o->name->str() == "Object") {
    middle("libbirch::Any");
  } else {
    middle("Object_");
  }
}
