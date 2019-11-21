/**
 * @file
 */
#include "bi/io/cpp/CppClassGenerator.hpp"

#include "bi/io/cpp/CppMemberFiberGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppClassGenerator::CppClassGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    type(nullptr) {
  //
}

void bi::CppClassGenerator::visit(const Class* o) {
  if (!o->isAlias() && o->isBound() && !o->braces->isEmpty()) {
    type = o;
    auto super = dynamic_cast<const ClassType*>(o->base->canonical());
    assert(o->base->isEmpty() || super);

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
      if (o->isGeneric() && o->isBound()) {
       genTemplateArgs(o);
      }
      if (o->has(FINAL)) {
        middle(" final");
      }
      if (!o->base->isEmpty()) {
        middle(" : public " << super->name);
        if (!super->typeArgs->isEmpty()) {
          middle('<' << super->typeArgs << '>');
        }
      } else {
        middle(" : public libbirch::Any");
      }
      finish(" {");
      line("public:");
      in();
      if (o->isBound()) {
        start("using class_type_ = " << o->name);
        genTemplateArgs(o);
        finish(';');
        line("using this_type_ = class_type_;");
        if (o->base->isEmpty()) {
          line("using super_type_ = libbirch::Any;");
        } else {
          start("using super_type_ = " << super->name);
          if (!super->typeArgs->isEmpty()) {
            middle('<' << super->typeArgs << '>');
          }
          middle(';');
        }
        line("");
        line("using super_type_::operator=;");
        line("");

        /* using declarations for member functions and fibers in base classes
         * that are overridden */
        std::set<std::string> names;
        for (auto f : memberFunctions) {
          if (o->scope->override(f)) {
            names.insert(f->name->str());
          }
        }
        for (auto f : memberFibers) {
          if (o->scope->override(f)) {
            names.insert(f->name->str());
          }
        }
        for (auto name : names) {
          line("using super_type_::" << internalise(name) << ';');
        }
        line("");
      }
    }

    /* constructor */
    if (!header) {
      genTraceLine(o->loc);
      start("bi::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
    } else {
      start("");
    }
    middle(o->name << "(libbirch::Label* context_");
    if (!o->params->isEmpty()) {
      CppBaseGenerator aux(base, level, header);
      aux << ", " << o->params;
    }
    middle(')');
    if (header) {
      finish(";\n");
    } else {
      finish(" :");
      in();
      in();
      genTraceLine(o->loc);
      start("super_type_(context_");
      if (!o->args->isEmpty()) {
        middle(", " << o->args);
      }
      middle(')');
      ++inConstructor;
      for (auto o : memberVariables) {
        if (!o->value->isEmpty()) {
          finish(',');
          genTraceLine(o->loc);
          start(o->name << '(');
          if (!o->type->isValue()) {
            middle("context_, ");
          }
          middle(o->value << ')');
        } else if (o->type->isClass()) {
          finish(',');
          genTraceLine(o->loc);
          start(o->name << "(context_, libbirch::make_pointer<" << o->type << ">(context_");
          if (!o->args->isEmpty()) {
            middle(", " << o->args);
          }
          middle("))");
        } else if (o->type->isArray() && !o->brackets->isEmpty()) {
          finish(',');
          genTraceLine(o->loc);
          start(o->name << '(');
          if (!o->type->isValue()) {
            middle("context_, ");
          }
          middle("libbirch::make_shape(" << o->brackets << ')');
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
      genTraceLine(o->loc);
      start("bi::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
    } else {
      start("");
    }
    middle(o->name << "(libbirch::Label* context, libbirch::Label* label, const " << o->name << "& o)");
    if (header) {
      finish(";\n");
    } else {
      finish(" :");
      in();
      in();
      genTraceLine(o->loc);
      start("super_type_(context, label, o)");
      for (auto o : memberVariables) {
        finish(',');
        genTraceLine(o->loc);
        if (o->type->isValue()) {
          start(o->name << "(o." << o->name << ')');
        } else {
          start(o->name << "(context, label, o." << o->name << ')');
        }
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
      line("virtual ~" << o->name << "() = default;  // LCOV_EXCL_LINE");
      line(o->name << "(const " << o->name << "&) = delete;");
      line(o->name << "& operator=(const " << o->name << "&) = delete;");
    }

    /* clone function */
    if (!o->has(ABSTRACT)) {
      if (header) {
        line("virtual " << o->name << "* clone_(libbirch::Label* context_) const {");
        in();
        line("return libbirch::clone_object<" << o->name << ">(context_, this);  // LCOV_EXCL_LINE");
        out();
        line("}\n");
      }
    }

    /* name function */
    if (header) {
      line("virtual const char* getClassName() const {");
      in();
      line("return \"" << o->name << "\";  // LCOV_EXCL_LINE");
      out();
      line("}\n");
    }

    /* freeze function */
    if (header) {
      start("virtual void ");
    } else {
      genTraceLine(o->loc);
      start("void bi::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
    }
    middle("doFreeze_()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceLine(o->loc);
      line("super_type_::doFreeze_();");
      for (auto o : memberVariables) {
        if (!o->type->isValue()) {
          genTraceLine(o->loc);
          line(o->name << ".freeze();");
        }
      }
      out();
      line("}\n");
    }

    /* thaw function */
    if (header) {
      start("virtual void ");
    } else {
      genTraceLine(o->loc);
      start("void bi::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
    }
    middle("doThaw_(libbirch::Label* label_)");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceLine(o->loc);
      line("super_type_::doThaw_(label_);");
      for (auto o : memberVariables) {
        if (!o->type->isValue()) {
          genTraceLine(o->loc);
          line(o->name << ".thaw(label_);");
        }
      }
      out();
      line("}\n");
    }

    /* finish function */
    if (header) {
      start("virtual void ");
    } else {
      genTraceLine(o->loc);
      start("void bi::type::" << o->name);
      genTemplateArgs(o);
      middle("::");
    }
    middle("doFinish_()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceLine(o->loc);
      line("super_type_::doFinish_();");
      for (auto o : memberVariables) {
        if (!o->type->isValue()) {
          genTraceLine(o->loc);
          line(o->name << ".finish();");
        }
      }
      out();
      line("}");
    }

    /* setters for member variables */
    if (header) {
      Gatherer<MemberVariable> memberVars;
      o->accept(&memberVars);
      for (auto var : memberVars) {
        line("template<class T_>");
        line("auto& set_" << var->name << "_(T_&& o_) {");
        in();
        start("return " << var->name);
        if (var->type->isValue()) {
          middle(" = std::forward<T_>(o_)");
        } else {
          middle(".assign(this->getLabel(), std::forward<T_>(o_))");
        }
        finish(";  // LCOV_EXCL_LINE");
        out();
        line("}\n");

        if (var->type->isArray()) {
          line("template<class F_, class T_>");
          line("auto set_" << var->name << "_(const F_& shape_, T_&& o_) {");
          in();
          start("return " << var->name << ".get(shape_)");
          if (var->type->isValue()) {
            middle(" = std::forward<T_>(o_)");
          } else {
            middle(".assign(this->getLabel(), std::forward<T_>(o_))");
          }
          finish(";  // LCOV_EXCL_LINE");
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
      if (header) {
        start("extern \"C\" bi::type::" << o->name << "* ");
        finish("make_" << o->name << "_(libbirch::Label* context_);");
      } else {
        genTraceLine(o->loc);
        start("bi::type::" << o->name << "* ");
        finish("bi::type::make_" << o->name << "_(libbirch::Label* context_) {");
        in();
        genTraceLine(o->loc);
        line("return new bi::type::" << o->name << "(context_);");
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
  if (header || !o->braces->isEmpty()) {
    if (header) {
      start("virtual ");
    } else {
      genTraceLine(o->loc);
      start("");
    }
    middle(o->returnType << ' ');
    if (!header) {
      middle("bi::type::" << type->name);
      genTemplateArgs(type);
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

      /* swap context */
      genTraceLine(o->loc);
      line("libbirch_swap_context_");

      /* declare self if necessary */
      Gatherer<Member> members;
      Gatherer<Raw> raws;
      Gatherer<This> selfs;
      Gatherer<Super> supers;
      o->accept(&members);
      o->accept(&raws);
      o->accept(&selfs);
      o->accept(&supers);
      if (members.size() + raws.size() + selfs.size() + supers.size() > 0) {
        genTraceLine(o->loc);
        line("libbirch_declare_self_");
      }

      /* body */
      genTraceLine(o->loc);
      genTraceFunction(o->name->str(), o->loc);

      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();

      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const MemberFiber* o) {
  if (header || !o->braces->isEmpty()) {
    CppMemberFiberGenerator auxMemberFiber(type, base, level, header);
    auxMemberFiber << o;
  }
}

void bi::CppClassGenerator::visit(const AssignmentOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      start("virtual ");
    } else {
      genTraceLine(o->loc);
      start("bi::type::");
    }
    middle(type->name);
    genTemplateArgs(type);
    middle("& ");
    if (!header) {
      middle("bi::type::" << type->name);
      genTemplateArgs(type);
      middle("::");
    }
    middle("operator=(" << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceLine(o->loc);
      line("libbirch_swap_context_");
      genTraceLine(o->loc);
      line("libbirch_declare_self_");
      genTraceLine(o->loc);
      genTraceFunction("<assignment>", o->loc);
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      genTraceLine(o->loc);
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
      genTraceLine(o->loc);
      start("bi::type::" << type->name);
      genTemplateArgs(type);
      middle("::");
    }
    middle("operator " << o->returnType << "()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      genTraceLine(o->loc);
      line("libbirch_swap_context_");
      genTraceLine(o->loc);
      line("libbirch_declare_self_ ");
      genTraceLine(o->loc);
      genTraceFunction("<conversion>", o->loc);
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}
