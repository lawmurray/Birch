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
    label(0),
    inFor(false) {
  //
}

void bi::CppFiberGenerator::visit(const Fiber* o) {
  /* generate a unique name (within this scope) for the state of the fiber */
  std::stringstream base;
  bih_ostream buf(base);
  buf << o->params;
  std::string baseName = internalise(o->name->str()) + '_' + encode32(base.str());
  std::string stateName = baseName + "_FiberState";

  /* gather important objects */
  o->params->accept(&params);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  if (o->isBound()) {
    /* supporting class for state */
    if (header) {
      start("class " << stateName);
      finish(" : public FiberState<" << o->returnType->unwrap() << "> {");
      line("public:");
      in();
      line("using class_type = " << stateName << ';');
      line("using super_type = FiberState<" << o->returnType->unwrap() << ">;\n");
      for (auto param : params) {
        line(param->type << ' ' << param->name << ';');
      }
      for (auto local : locals) {
        start(local->type << ' ');
        finish(getName(local->name->str(), local->number) << ';');
      }
    }

    if (header) {
      out();
      line("protected:");
      in();
    }

    /* constructor */
    if (!header) {
      start("bi::" << stateName << "::");
    } else {
      start("");
    }
    middle(stateName << '(' << o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" :");
      in();
      in();
      start("super_type(0, " << (yields.size() + 1) << ')');
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

    /* copy constructor, destructor, assignment operator */
    if (header) {
      line(stateName << "(const " << stateName << "&) = default;");
      line("virtual ~" << stateName << "() = default;");
      line(stateName << "& operator=(const " << stateName << "&) = default;");
    }

    if (header) {
      out();
      line("public:");
      in();
    }

    /* standard functions */
    if (header) {
      line("STANDARD_CREATE_FUNCTION");
      line("STANDARD_EMPLACE_FUNCTION");
      line("STANDARD_CLONE_FUNCTION");
      line("STANDARD_DESTROY_FUNCTION");
    }

    /* freeze function */
    if (header) {
      start("virtual void ");
    } else {
      start("void bi::" << stateName << "::");
    }
    middle("doFreeze()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      line("super_type::doFreeze();");
      line("bi::freeze(value);");
      for (auto param : params) {
        line("bi::freeze(" << param->name << ");");
      }
      for (auto local : locals) {
        line("bi::freeze(" << getName(local->name->str(), local->number) << ");");
      }
      out();
      line("}\n");
    }

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
      line("SharedCOW<class_type> local(this, context.get());");
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
  }

  /* initialisation function */
  if (o->isGeneric()) {
    /* generic functions are generated as a struct with a static member
     * function, where the type parameters are part of the struct; this means
     * we don't have to generate even a signature for the unbound function */
    if (header) {
      genTemplateParams(o);
      start("struct " << o->name);
      if (o->isBound()) {
        genTemplateArgs(o);
      }
      finish(" {");
      in();
      if (o->isBound()) {
        start("static ");
      } else {
        line("//");
      }
    }
    if (o->isBound()) {
      middle(o->returnType << ' ');
      if (!header) {
        start("bi::" << o->name);
        genTemplateArgs(o);
        middle("::");
      }
      middle("f(" << o->params << ')');
      if (header) {
        finish(';');
      }
    }
    if (header) {
      out();
      line("};\n");
    }
  } else {
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    middle(o->name << '(' << o->params << ')');
    if (header) {
      finish(';');
    }
  }

  /* body */
  if (!header && o->isBound()) {
    finish(" {");
    in();
    start("return make_fiber<" << stateName << ">(");
    for (auto iter = params.begin(); iter != params.end(); ++iter) {
      if (iter != params.begin()) {
        middle(", ");
      }
      middle((*iter)->name);
    }
    finish(");");
    out();
    line("}\n");
  }
}

void bi::CppFiberGenerator::visit(const Return* o) {
  genTraceLine(o->loc->firstLine);
  line("goto END;");
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  genTraceLine(o->loc->firstLine);
  line("local->value = " << o->single << ';');
  line("local->label = " << label << ';');
  line("return true;");
  line("LABEL" << label << ": ;");
  ++label;
}

void bi::CppFiberGenerator::visit(const Identifier<Parameter>* o) {
  middle("local->" << o->name);
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
    middle("local->" << name << " = " << o->type << "::create()");
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
  line("switch (local->label) {");
  in();
  for (int s = 0; s <= yields.size(); ++s) {
    line("case " << s << ": goto LABEL" << s << ';');
  }
  line("default: goto END;");
  out();
  line('}');
  line("LABEL0:");
  ++label;
}

void bi::CppFiberGenerator::genEnd() {
  line("END:");
  line("local->label = " << (yields.size() + 1) << ';');
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
