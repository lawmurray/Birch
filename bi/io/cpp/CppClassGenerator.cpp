/**
#include <bi/io/cpp/CppFiberGenerator.hpp>
 * @file
 */
#include "bi/io/cpp/CppClassGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

bi::CppClassGenerator::CppClassGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    theClass(nullptr) {
  //
}

void bi::CppClassGenerator::visit(const Class* o) {
  if (!o->isAlias() && !o->braces->isEmpty()) {
    theClass = o;
    auto base = dynamic_cast<const NamedType*>(o->base);

    Gatherer<MemberFunction> memberFunctions;
    Gatherer<MemberFiber> memberFibers;
    Gatherer<MemberVariable> memberVariables;
    o->accept(&memberFunctions);
    o->accept(&memberFibers);
    o->accept(&memberVariables);

    /* start boilerplate */
    if (header) {
      genTemplateParams(o);
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
      start("using class_type_ = " << o->name);
      genTemplateArgs(o);
      finish(';');
      line("using this_type_ = class_type_;");
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

      line("using super_type_::operator=;");
      for (auto name : names) {
        line("using super_type_::" << internalise(name) << ';');
      }
      line("");
    }

    /* constructor */
    if (!header) {
      genSourceLine(o->loc);
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("bi::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
    } else {
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
        if (!o->value->isEmpty()) {
          finish(',');
          genSourceLine(o->loc);
          start(o->name << '(' << o->value << ')');
        } else if (o->type->isClass()) {
          finish(',');
          genSourceLine(o->loc);
          start(o->name << "(libbirch::make_pointer<" << o->type << ">(");
          middle(o->args << "))");
        } else if (o->type->isArray() && !o->brackets->isEmpty()) {
          finish(',');
          genSourceLine(o->loc);
          start(o->name << "(libbirch::make_shape(" << o->brackets << ')');
          if (!o->args->isEmpty()) {
            middle(", " << o->args);
          }
          middle(')');
        }
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

    /* deep copy constructor */
    if (!header) {
      genSourceLine(o->loc);
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("bi::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
    } else {
      start("");
    }
    middle(o->name << "(libbirch::Label* label, const " << o->name << "& o)");
    if (header) {
      finish(";\n");
    } else {
      finish(" :");
      in();
      in();
      genSourceLine(o->loc);
      start("super_type_(label, o)");
      for (auto o : memberVariables) {
        finish(',');
        genSourceLine(o->loc);
        start(o->name << "(libbirch::clone(label, o." << o->name << "))");
      }
      out();
      out();
      finish(" {");
      in();
      line("//");
      out();
      line("}\n");
    }

    /* copy constructor, destructor, assignment operator */
    if (header) {
      line("virtual ~" << o->name << "() = default;");
      line(o->name << "(const " << o->name << "&) = delete;");
      line(o->name << "& operator=(const " << o->name << "&) = delete;");
    }

    /* clone function */
    if (!o->has(ABSTRACT)) {
      if (header) {
        line("virtual " << o->name << "* clone_(libbirch::Label* label) const;");
      } else {
        genSourceLine(o->loc);
        genTemplateParams(o);
        genSourceLine(o->loc);
        start("bi::type::" << o->name);
        genTemplateArgs(o);
        middle("* bi::type::" << o->name);
        genTemplateArgs(o);
        middle("::");
        finish("clone_(libbirch::Label* label) const {");
        in();
        genSourceLine(o->loc);
        line("return new class_type_(label, *this);");
        genSourceLine(o->loc);
        out();
        line("}\n");
      }
    }

    /* name function */
    if (header) {
      line("virtual const char* getClassName() const;");
    } else {
      genSourceLine(o->loc);
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("const char* bi::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
      finish("getClassName() const {");
      in();
      genSourceLine(o->loc);
      line("return \"" << o->name << "\";");
      genSourceLine(o->loc);
      out();
      line("}\n");
    }

    /* freeze function */
    if (header) {
      line("virtual void doFreeze_();");
    } else {
      genSourceLine(o->loc);
      genTemplateParams(o);
      start("void bi::type::" << o->name);
      genTemplateArgs(o);
      finish("::doFreeze_() {");
      in();
      genSourceLine(o->loc);
      line("super_type_::doFreeze_();");
      for (auto o : memberVariables) {
        genSourceLine(o->loc);
        line("freeze(" << o->name << ");");
      }
      genSourceLine(o->loc);
      out();
      line("}\n");
    }

    /* thaw function */
    if (header) {
      line("virtual void doThaw(libbirch::Label* label_);");
    } else {
      genSourceLine(o->loc);
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("void bi::type::" << o->name);
      genTemplateArgs(o);
      finish("::doThaw(libbirch::Label* label_) {");
      in();
      genSourceLine(o->loc);
      line("super_type_::doThaw_(label_);");
      for (auto o : memberVariables) {
        genSourceLine(o->loc);
        line("thaw(" << o->name << ", label_);");
      }
      genSourceLine(o->loc);
      out();
      line("}\n");
    }

    /* finish function */
    if (header) {
      line("virtual void doFinish_();");
    } else {
      genSourceLine(o->loc);
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("void bi::type::" << o->name);
      genTemplateArgs(o);
      finish("::doFinish_() {");
      in();
      genSourceLine(o->loc);
      line("super_type_::doFinish_();");
      for (auto o : memberVariables) {
        genSourceLine(o->loc);
        line("finish(" << o->name << ");");
      }
      genSourceLine(o->loc);
      out();
      line("}\n");
    }

    /* setters for member variables */
    if (header) {
      Gatherer<MemberVariable> memberVars;
      o->accept(&memberVars);
      for (auto var : memberVars) {
        genSourceLine(var->loc);
        line("template<class T_> auto& set_" << var->name << "_(T_&& o_) {");
        in();
        genSourceLine(var->loc);
        line("return " << var->name << " = std::forward<T_>(o_);");
        genSourceLine(var->loc);
        out();
        line("}\n");

        if (var->type->isArray()) {
          genSourceLine(var->loc);
          line("template<class F_, class T_> auto set_" << var->name << "_(const F_& shape_, T_&& o_) {");
          in();
          genSourceLine(var->loc);
          line("return " << var->name << ".get(shape_) = std::forward<T_>(o_);");
          genSourceLine(var->loc);
          out();
          line("}\n");
        }
      }
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
      start("virtual ");
    } else {
      genSourceLine(o->loc);
      genTemplateParams(theClass);
      genSourceLine(o->loc);
      start("");
    }
    middle(o->returnType << ' ');
    if (!header) {
      middle("bi::type::" << theClass->name);
      genTemplateArgs(theClass);
      middle("::");
    }
    middle(internalise(o->name->str()) << '(' << o->params << ')');
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
      line("libbirch_declare_self_");
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const MemberFiber* o) {
  if ((header && o->has(ABSTRACT)) || !o->braces->isEmpty()) {
    if (header) {
      start("virtual ");
    } else {
      genSourceLine(o->loc);
      genTemplateParams(theClass);
      genSourceLine(o->loc);
      start("");
    }
    middle(o->returnType << ' ');
    if (!header) {
      middle("bi::type::" << theClass->name);
      genTemplateArgs(theClass);
      middle("::");
    }
    middle(internalise(o->name->str()) << '(' << o->params << ')');
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
      line("libbirch_declare_self_");
      //CppResumeGenerator aux(nullptr, base, level, header);
      //aux << o->yield;
      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const AssignmentOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      start("virtual ");
    } else {
      genSourceLine(o->loc);
      genTemplateParams(theClass);
      genSourceLine(o->loc);
      start("bi::type::");
    }
    middle(theClass->name);
    genTemplateArgs(theClass);
    middle("& ");
    if (!header) {
      middle("bi::type::" << theClass->name);
      genTemplateArgs(theClass);
      middle("::");
    }
    middle("operator=(" << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genSourceLine(o->loc);
      line("libbirch_declare_self_");
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
      start("virtual ");
    } else {
      genSourceLine(o->loc);
      genTemplateParams(theClass);
      genSourceLine(o->loc);
      start("bi::type::" << theClass->name);
      genTemplateArgs(theClass);
      middle("::");
    }
    middle("operator " << o->returnType << "()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genSourceLine(o->loc);
      line("libbirch_declare_self_ ");
      genTraceFunction("<conversion>", o->loc);
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}
