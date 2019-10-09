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
      finish("public libbirch::FiberState<" << o->returnType->unwrap() << "> {");
      line("public:");
      in();
      line("using class_type_ = " << stateName << ';');
      line("using super_type_ = libbirch::FiberState<" << o->returnType->unwrap() << ">;\n");
      for (auto param : params) {
        line(param->type << ' ' << param->name << ';');
      }
      for (auto local : locals) {
        start(local->type << ' ');
        finish(getName(local->name->str(), local->number) << ';');
      }
    }

    /* constructor */
    if (!header) {
      start("bi::" << stateName << "::");
    } else {
      start("");
    }
    middle(stateName << "(libbirch::Label* label_, " << o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" :");
      in();
      in();
      start("super_type_(label_, " << (yields.size() + 1) << ')');
      for (auto param : params) {
        finish(',');
        start(param->name << '(' << param->name << ')');
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
    middle(stateName << "(libbirch::Label* label_, const " << stateName << "& o_)");
    if (header) {
      finish(";\n");
    } else {
      finish(" :");
      in();
      in();
      start("super_type_(label_, o_)");
      for (auto param : params) {
        if (!param->type->isValue()) {
          auto name = param->name;
          finish(',');
          start(name << "(label_, o_." << name << ')');
        }
      }
      for (auto local : locals) {
        if (!local->type->isValue()) {
          auto name = getName(local->name->str(), local->number);
          finish(',');
          start(name << "(label_, o_." << name << ')');
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
      line("virtual " << stateName << "* clone_(libbirch::Label* label_) const {");
      in();
      line("return new " << stateName << "(label_, *this);");
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
      line("libbirch::freeze(value_);");
      for (auto param : params) {
        line("libbirch::freeze(" << param->name << ");");
      }
      for (auto local : locals) {
        line("libbirch::freeze(" << getName(local->name->str(), local->number) << ");");
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
    middle("doThaw_(libbirch::LazyLabel* label_)");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      line("super_type_::doThaw_(label_);");
      line("libbirch::thaw(value_, label_);");
      for (auto param : params) {
        line("libbirch::thaw(" << param->name << ", label_);");
      }
      for (auto local : locals) {
        line("libbirch::thaw(" << getName(local->name->str(), local->number) << ", label_);");
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
      line("libbirch::finish(value_);");
      for (auto param : params) {
        line("libbirch::finish(" << param->name << ");");
      }
      for (auto local : locals) {
        line("libbirch::finish(" << getName(local->name->str(), local->number) << ");");
      }
      out();
      line("}");
    }
    line("#endif\n");

    /* query function */
    if (header) {
      start("virtual ");
    } else {
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
      genTraceFunction(o->name->str(), o->loc);
      line("libbirch_swap_context_");
      line("libbirch_declare_local_");
      genSwitch();
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
    middle(name << "(libbirch::Label* context_");
    if (!o->params->isEmpty()) {
      middle(", " << o->params);
    }
    middle(')');
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
  }
}

void bi::CppFiberGenerator::visit(const Return* o) {
  genTraceLine(o->loc->firstLine);
  if (inLambda) {
    line("return " << o->single << ';');
  } else {
    line("goto END_;");
  }
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  genTraceLine(o->loc->firstLine);
  line("local->value_ = " << o->single << ';');
  line("local->point_ = " << point << ';');
  line("return true;");
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
  if (inFor || !o->value->isEmpty() || !o->args->isEmpty()
      || !o->brackets->isEmpty()) {
    /* the variable is declared in the fiber state, so there is no need to
     * do anything here unless there is some initialization associated with
     * it */
    inFor = false;
    middle("local->" << getName(o->name->str(), o->number));
    genInit(o);
  } else if (o->type->isClass()) {
    /* make sure objects are initialized, not just null pointers */
    auto name = getName(o->name->str(), o->number);
    ++inPointer;
    middle("local->" << name << " = libbirch::make_object<" << o->type << ">(context_)");
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
  line("END_:");
  line("local->point_ = " << (yields.size() + 1) << ';');
  line("return false;");
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
