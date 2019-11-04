/**
 * @file
 */
#include "bi/io/cpp/CppFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

bi::CppFiberGenerator::CppFiberGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    point(0),
    inFor(false) {
  //
}

void bi::CppFiberGenerator::visit(const Fiber* o) {
  yieldType = o->returnType->unwrap();

  /* generate a unique name (within this scope) for the state of the fiber */
  std::stringstream base;
  bih_ostream buf(base);
  buf << o->params;
  std::string baseName = internalise(o->name->str()) + '_' + encode32(base.str());
  std::string stateName = baseName + "_FiberState_";

  /* gather important objects */
  o->params->accept(&params);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  if (o->isBound()) {
    /* supporting class for state */
    if (header) {
      start("class " << stateName << " final : ");
      finish("public libbirch::FiberState<" << yieldType << "> {");
      line("public:");
      in();
      line("using class_type_ = " << stateName << ';');
      line("using super_type_ = libbirch::FiberState<" << yieldType << ">;\n");
      for (auto param : params) {
        line(param->type << ' ' << param->name << ';');
      }
      for (auto local : locals) {
        start(local->type << ' ');
        finish(getName(local->name->str(), local->number) << ';');
      }
    }

    line("// LCOV_EXCL_START");

    /* constructor */
    if (!header) {
      start("bi::" << stateName << "::");
    } else {
      start("");
    }
    middle(stateName << "(libbirch::Label* context_, " << o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" :");
      in();
      in();
      start("super_type_(context_, " << (yields.size() + 1) << ')');
      for (auto param : params) {
        finish(',');
        start(param->name << '(');
        if (!param->type->isValue()) {
          middle("context_, ");
        }
        middle(param->name << ')');
      }
      finish(" {");
      out();
      line("//");
      out();
      line("}\n");
    }

    /* deep copy constructor */
    if (!header) {
      start("bi::" << stateName << "::");
    } else {
      start("");
    }
    middle(stateName << "(libbirch::Label* context, libbirch::Label* label, const " << stateName << "& o)");
    if (header) {
      finish(";\n");
    } else {
      finish(" :");
      in();
      in();
      start("super_type_(context, label, o)");
      for (auto o : params) {
        if (!o->type->isValue()) {
          finish(',');
          if (o->type->isValue()) {
            start(o->name << "(o." << o->name << ')');
          } else {
            start(o->name << "(context, label, o." << o->name << ')');
          }
        }
      }
      for (auto o : locals) {
        if (!o->type->isValue()) {
          auto name = getName(o->name->str(), o->number);
          finish(',');
          if (o->type->isValue()) {
            start(name << "(o." << name << ')');
          } else {
            start(name << "(context, label, o." << name << ')');
          }
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
      line("virtual ~" << stateName << "() = default;");
      line(stateName << "(const " << stateName << "&) = delete;");
      line(stateName << "& operator=(const " << stateName << "&) = delete;");
    }

    /* clone function */
    if (header) {
      line("virtual " << stateName << "* clone_(libbirch::Label* context_) const {");
      in();
      line("return libbirch::clone_object<" << stateName << ">(context_, this);");
      out();
      line("}\n");
    }

    /* name function */
    if (header) {
      line("virtual const char* name_() const {");
      in();
      line("return \"" << stateName << "\";");
      out();
      line("}\n");
    }

    /* freeze function */
    line("#if ENABLE_LAZY_DEEP_CLONE");
    if (header) {
      start("virtual void ");
    } else {
      start("void bi::" << stateName << "::");
    }
    middle("doFreeze_()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      line("super_type_::doFreeze_();");
      if (!o->returnType->unwrap()->isValue()) {
        line("value_.freeze();");
      }
      for (auto o : params) {
        if (!o->type->isValue()) {
          line(o->name << ".freeze();");
        }
      }
      for (auto o : locals) {
        if (!o->type->isValue()) {
          line(getName(o->name->str(), o->number) << ".freeze();");
        }
      }
      out();
      line("}\n");
    }

    /* thaw function */
    if (header) {
      start("virtual void ");
    } else {
      start("void bi::" << stateName << "::");
    }
    middle("doThaw_(libbirch::Label* label_)");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      line("super_type_::doThaw_(label_);");
      if (!o->returnType->unwrap()->isValue()) {
        line("value_.thaw(label_);");
      }
      for (auto o : params) {
        if (!o->type->isValue()) {
          line(o->name << ".thaw(label_);");
        }
      }
      for (auto o : locals) {
        if (!o->type->isValue()) {
          line(getName(o->name->str(), o->number) << ".thaw(label_);");
        }
      }
      out();
      line("}\n");
    }

    /* finish function */
    if (header) {
      start("virtual void ");
    } else {
      start("void bi::" << stateName << "::");
    }
    middle("doFinish_()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      line("super_type_::doFinish_();");
      if (!o->returnType->unwrap()->isValue()) {
        line("value_.finish();");
      }
      for (auto o : params) {
        if (!o->type->isValue()) {
          line(o->name << ".finish();");
        }
      }
      for (auto o : locals) {
        if (!o->type->isValue()) {
          line(getName(o->name->str(), o->number) << ".finish();");
        }
      }
      out();
      line("}");
    }
    line("#endif\n");

    line("// LCOV_EXCL_STOP");

    /* query function */
    if (header) {
      start("virtual ");
    } else {
      genTraceLine(o->loc);
      start("");
    }
    middle("bool ");
    if (!header) {
      middle("bi::" << stateName << "::");
    }
    middle("query()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      for (auto iter = o->typeParams->begin(); iter != o->typeParams->end();
          ++iter) {
        auto param = dynamic_cast<const Generic*>(*iter);
        assert(param);
        line("using " << param->name << " [[maybe_unused]] = " << param->type << ';');
      }
      line("// LCOV_EXCL_START");
      line("libbirch_swap_context_");
      line("libbirch_declare_local_");
      genTraceFunction(o->name->str(), o->loc);
      genSwitch();
      line("// LCOV_EXCL_STOP");
      *this << o->braces->strip();
      genEnd();
      out();
      finish("}\n");
    }
    if (header) {
      out();
      line("};\n");
    }

    /* initialisation function */
    line("// LCOV_EXCL_START");
    auto name = internalise(o->name->str());
    if (o->isInstantiation()) {
      std::stringstream base;
      bih_ostream buf(base);
      buf << o->typeParams << '(' << o->params->type << ')';
      name += "_" + encode32(base.str()) + "_";
    }
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    middle(name << '(');
    if (!o->isValue()) {
      middle("libbirch::Label* context_");
      if (!o->params->isEmpty()) {
        middle(", ");
      }
    }
    middle(o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      start("return libbirch::make_fiber<" << stateName << ">(context_");
      for (auto iter = params.begin(); iter != params.end(); ++iter) {
        middle(", " << (*iter)->name);
      }
      finish(");");
      out();
      line("}\n");
    }
    line("// LCOV_EXCL_STOP");
  }
}

void bi::CppFiberGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  if (inLambda) {
    line("return " << o->single << ';');
  } else {
    line("goto END_;");
  }
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  genTraceLine(o->loc);
  start("local->value_");
  if (yieldType->isValue()) {
    finish(" = " << o->single << ';');
  } else {
    finish(".assign(context_, " << o->single << ");");
  }
  genTraceLine(o->loc);
  line("local->point_ = " << point << ';');
  genTraceLine(o->loc);
  line("return true;");
  genTraceLine(o->loc);
  line("POINT" << point << "_: ;");
  ++point;
}

void bi::CppFiberGenerator::visit(const Identifier<Parameter>* o) {
  if (!inLambda) {
    middle("local->");
  }
  middle(o->name);
}

void bi::CppFiberGenerator::visit(const Identifier<LocalVariable>* o) {
  middle("local->" << getName(o->name->str(), o->target->number));
}

void bi::CppFiberGenerator::visit(const LocalVariable* o) {
  auto name = getName(o->name->str(), o->number);
  if (inFor) {
    inFor = false;
    middle("local->" << name);
  } else {
    if (o->type->isValue()) {
      if (!o->value->isEmpty()) {
        middle("local->" << name << " = " << o->value);
      } else if (!o->brackets->isEmpty()) {
        middle("local->" << name << ".assign(");
        middle(o->type << "(libbirch::make_shape(" << o->brackets << "))");
        middle(')');
      }
    } else {
      if (!o->value->isEmpty()) {
        middle("local->" << name << ".assign(context_, ");
        middle(o->type << "(context_, " << o->value << "))");
      } else if (!o->args->isEmpty()) {
        middle("local->" << name << ".assign(context_, ");
        middle(o->type << "(context_, " << o->args << "))");
      } else if (!o->brackets->isEmpty()) {
        middle("local->" << name << ".assign(context_, ");
        middle(o->type << "(context_, " << o->brackets << "))");
      } else if (o->type->isClass()) {
        middle("local->" << name << ".assign(context_, ");
        middle("libbirch::make_pointer<" << o->type << ">(context_))");
      }
    }
  }
}

void bi::CppFiberGenerator::visit(const For* o) {
  /* special exemption for the handling of local variable initialisation
   * above: do need to initialise a local variable when it is declared as the
   * index of a for loop; the inFor flag is switched off after the first
   * local variable encountered */
  inFor = true;
  CppBaseGenerator::visit(o);
}

void bi::CppFiberGenerator::genSwitch() {
  line("switch (local->point_) {");
  in();
  for (int s = 0; s <= yields.size(); ++s) {
    line("case " << s << ": goto POINT" << s << "_;");
  }
  line("default: goto END_;");
  out();
  line('}');
  line("POINT0_:");
  ++point;
}

void bi::CppFiberGenerator::genEnd() {
  line("// LCOV_EXCL_START");
  line("END_:");
  line("local->point_ = " << (yields.size() + 1) << ';');
  line("return false;");
  line("// LCOV_EXCL_STOP");
}

std::string bi::CppFiberGenerator::getName(const std::string& name,
    const int number) {
  std::stringstream buf;
  std::string result;
  auto iter = names.find(number);
  if (iter == names.end()) {
    auto count = counts.find(name);
    if (count == counts.end()) {
      buf << internalise(name);
      result = buf.str();
      counts.insert(std::make_pair(name, 1));
    } else {
      buf << internalise(name) << count->second << '_';
      result = buf.str();
      ++count->second;
    }
    names.insert(std::make_pair(number, result));
  } else {
    result = iter->second;
  }
  return result;
}
