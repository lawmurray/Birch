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

  /* supporting class for state */
  if (header) {
    genTemplateParams(o);
    start("class " << stateName);
    if (o->isBound()) {
      genTemplateArgs(o);
    }
    finish(" : public FiberState<" << o->returnType->unwrap() << "> {");
    line("public:");
    in();
    line("using super_type = FiberState<" << o->returnType->unwrap() << ">;\n");
    for (auto param : params) {
      line(param->type << ' ' << param->name << ';');
    }
    for (auto local : locals) {
      start(local->type << ' ');
      finish(getName(local->name->str(), local->number) << ';');
    }
  }

  /* constructor */
  if (!header && !o->isBound()) {
    genTemplateParams(o);
  }
  if (!header) {
    start("bi::" << stateName);
    genTemplateArgs(o);
    middle("::");
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

  /* clone function */
  if (!header && !o->isBound()) {
    genTemplateParams(o);
  } else {
    start("");
  }
  if (header) {
    middle("virtual ");
  }
  middle("bi::FiberState<");
  middle(o->returnType->unwrap() << ">* ");
  if (!header) {
    middle("bi::" << stateName);
    genTemplateArgs(o);
    middle("::");
  }
  middle("clone() const");
  if (header) {
    finish(";\n");
  } else {
    finish(" {");
    in();
    line("return bi::construct<" << stateName << ">(*this);");
    out();
    line("}\n");
  }

  /* destroy function */
  if (!header && !o->isBound()) {
    genTemplateParams(o);
  } else {
    start("");
  }
  if (header) {
    middle("virtual ");
  }
  middle("void ");
  if (!header) {
    middle("bi::" << stateName);
    genTemplateArgs(o);
    middle("::");
  }
  middle("destroy()");
  if (header) {
    finish(";\n");
  } else {
    finish(" {");
    in();
    line("this->size = sizeof(*this);");
    line("this->~" << stateName << "();");
    out();
    line("}\n");
  }

  /* query function */
  if (!header && !o->isBound()) {
    genTemplateParams(o);
  } else {
    start("");
  }
  if (header) {
    middle("virtual ");
  }
  middle("bool ");
  if (!header) {
    middle("bi::" << stateName);
    genTemplateArgs(o);
    middle("::");
  }
  middle("query()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    line("Enter enter(getMemo());");
    genSwitch();
    *this << o->braces->strip();
    genEnd();
    out();
    finish("}\n");
  }
  if (header) {
    line("};\n");
  }

  /* initialisation function */
  genTemplateParams(o);
  start(o->returnType << ' ');
  if (!header) {
    middle("bi::");
  }
  middle(o->name);
  if (!header && o->isBound()) {
    genTemplateArgs(o);
  }
  middle('(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return make_fiber<" << stateName);
    genTemplateArgs(o);
    middle(">(");
    for (auto iter = params.begin(); iter != params.end(); ++iter) {
      if (iter != params.begin()) {
        middle(", ");
      }
      middle((*iter)->name);
    }
    finish(");");
    out();
    finish("}\n");
  }
}

void bi::CppFiberGenerator::visit(const Return* o) {
  genTraceLine(o->loc->firstLine);
  line("goto END;");
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  genTraceLine(o->loc->firstLine);
  line("this->value = " << o->single << ';');
  line("this->label = " << label << ';');
  line("return true;");
  line("LABEL" << label << ": ;");
  ++label;
}

void bi::CppFiberGenerator::visit(const Identifier<Parameter>* o) {
  if (!inMember) {
    middle("this->");
  }
  middle(o->name);
}

void bi::CppFiberGenerator::visit(const Identifier<LocalVariable>* o) {
  if (!inMember) {
    middle("this->");
  }
  middle(getName(o->name->str(), o->target->number));
}

void bi::CppFiberGenerator::visit(const LocalVariable* o) {
  if (inFor || !o->value->isEmpty() || !o->args->isEmpty()
      || !o->brackets->isEmpty()) {
    /* the variable is declared in the fiber state, so there is no need to
     * do anything here unless there is some initialization associated with
     * it */
    inFor = false;
    middle("this->" << getName(o->name->str(), o->number));
    genInit(o);
  } else if (o->type->isPointer() && !o->type->isWeak()) {
    /* make sure objects are initialized, not just null pointers */
    auto name = getName(o->name->str(), o->number);
    middle("this->" << name << " = bi::construct<" << o->type->unwrap() << ">()");
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
  line("switch (this->label) {");
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
  line("this->label = " << (yields.size() + 1) << ';');
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
