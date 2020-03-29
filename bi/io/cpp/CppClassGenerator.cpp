/**
 * @file
 */
#include "bi/io/cpp/CppClassGenerator.hpp"

#include "bi/io/cpp/CppResumeGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

bi::CppClassGenerator::CppClassGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    currentClass(nullptr) {
  //
}

void bi::CppClassGenerator::visit(const Class* o) {
  if (!o->isAlias() && !o->braces->isEmpty()) {
    currentClass = o;
    auto base = dynamic_cast<const NamedType*>(o->base);

    Gatherer<MemberFunction> memberFunctions;
    Gatherer<MemberFiber> memberFibers;
    Gatherer<MemberVariable> memberVariables;
    o->accept(&memberFunctions);
    o->accept(&memberFibers);
    o->accept(&memberVariables);

    if (header) {
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("class " << o->name);
      if (o->has(FINAL)) {
        middle(" final");
      }
      middle(" : public ");
      if (base) {
        middle(base->name);
        if (!base->typeArgs->isEmpty()) {
          middle('<' << base->typeArgs << '>');
        }
      } else {
        middle("libbirch::Any");
      }
      finish(" {");
      line("public:");
      in();
      genSourceLine(o->loc);
      start("using class_type_ = " << o->name);
      genTemplateArgs(o);
      finish(';');
      genSourceLine(o->loc);
      line("using this_type_ = class_type_;");
      genSourceLine(o->loc);
      start("using super_type_ = ");
      if (base) {
        middle(base->name);
        if (!base->typeArgs->isEmpty()) {
          middle('<' << base->typeArgs << '>');
        }
      } else {
        middle("libbirch::Any");
      }
      finish(";\n");

      /* using declarations for member functions and fibers in base classes
       * that are overridden */
      std::set<std::string> names;
      for (auto f : memberFunctions) {
        auto name = f->name->str();
        if (o->scope->overrides(name)) {
          names.insert(name);
        }
      }
      for (auto f : memberFibers) {
        auto name = f->name->str();
        if (o->scope->overrides(name)) {
          names.insert(name);
        }
      }

      genSourceLine(o->loc);
      line("using super_type_::operator=;");
      for (auto name : names) {
        genSourceLine(o->loc);
        line("using super_type_::" << internalise(name) << ';');
      }
      line("");
    }

    /* boilerplate */
    if (header) {
    	if (o->has(ABSTRACT)) {
    	  start("LIBBIRCH_ABSTRACT_CLASS");
    	} else {
    	  start("LIBBIRCH_CLASS");
    	}
      middle('(' << o->name << ", ");
      if (base) {
        middle(base->name);
        if (!base->typeArgs->isEmpty()) {
          middle('<' << base->typeArgs << '>');
        }
      } else {
        middle("libbirch::Any");
      }
      finish(')');
      start("LIBBIRCH_MEMBERS(");
      for (auto iter = memberVariables.begin(); iter != memberVariables.end();
          ++iter) {
        if (iter != memberVariables.begin()) {
          middle(", ");
        }
        middle((*iter)->name);
      }
      finish(")\n");
    }

    /* constructor */
    if (!header) {
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("bi::type::" << o->name);
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
      start("super_type_(" << o->args << ')');
      ++inConstructor;
      for (auto o : memberVariables) {
	      finish(',');
        genSourceLine(o->loc);
        start(o->name << '(');
        if (!o->value->isEmpty()) {
          middle(o->value);
        } else if (!o->brackets->isEmpty()) {
          middle("libbirch::make_shape(" << o->brackets << ')');
          if (!o->args->isEmpty()) {
            middle(", " << o->args);
          }
        } else if (!o->args->isEmpty()) {
          middle(o->args);
				}
        middle(')');
      }
      --inConstructor;
      out();
      out();
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
        start("extern \"C\" bi::type::" << o->name << "* ");
        finish("make_" << o->name << "_();");
      } else {
        start("bi::type::" << o->name << "* ");
        finish("bi::type::make_" << o->name << "_() {");
        in();
        genSourceLine(o->loc);
        line("return new bi::type::" << o->name << "();");
        genSourceLine(o->loc);
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
  if ((header && o->has(ABSTRACT)) || !o->braces->isEmpty()) {
    if (header) {
      genSourceLine(o->loc);
      start("virtual ");
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start("");
    }
    middle(o->returnType << ' ');
    if (!header) {
      middle("bi::type::" << currentClass->name);
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle(o->name << '(' << o->params << ')');
    if (header) {
      if (o->has(FINAL)) {
        middle(" final");
      } else if (o->has(ABSTRACT)) {
        middle(" = 0");
      }
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const MemberFiber* o) {
  if ((header && o->has(ABSTRACT)) || !o->braces->isEmpty()) {
    /* initialization function */
    if (header) {
      genSourceLine(o->loc);
      start("virtual ");
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start("");
    }
    middle(o->returnType << ' ');
    if (!header) {
      middle("bi::type::" << currentClass->name);
      genTemplateArgs(currentClass);
      middle("::");
    }
    middle(o->name << '(' << o->params << ')');
    if (header) {
      if (o->has(FINAL)) {
        middle(" final");
      } else if (o->has(ABSTRACT)) {
        middle(" = 0");
      }
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      genTraceLine(o->loc);
      line("yield_" << currentClass->name << '_' << o->name << '_' << o->number << "_0_();");
      out();
      line("}\n");
    }

    /* start function */
    CppResumeGenerator auxResume(currentClass, o, base, level, header);
    auxResume << o->start;

    /* resume functions */
    Gatherer<Yield> yields;
    o->accept(&yields);
    for (auto yield : yields) {
      if (yield->resume) {
        CppResumeGenerator auxResume(currentClass, o, base, level, header);
        auxResume << yield->resume;
      }
    }
  }
}

void bi::CppClassGenerator::visit(const AssignmentOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      genSourceLine(o->loc);
      start("virtual ");
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start("bi::type::");
    }
    middle(currentClass->name);
    genTemplateArgs(currentClass);
    middle("& ");
    if (!header) {
      middle("bi::type::" << currentClass->name);
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
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      genSourceLine(o->loc);
      line("return *this;");
      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const ConversionOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      genSourceLine(o->loc);
      start("virtual ");
    } else {
      genTemplateParams(currentClass);
      genSourceLine(o->loc);
      start("bi::type::" << currentClass->name);
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
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}
