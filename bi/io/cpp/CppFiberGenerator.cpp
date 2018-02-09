/**
 * @file
 */
#include "bi/io/cpp/CppFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

#include <sstream>

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
  std::string stateName = o->name->str() + '_' + encode32(base.str())
      + "_FiberState";

  /* gather important objects */
  o->params->accept(&parameters);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class */
  if (header) {
    start("class " << stateName << " : ");
    finish("public GlobalFiberState<" << o->returnType->unwrap() << "> {");
    line("public:");
    in();
    start("using super_type = GlobalFiberState<");
    middle(o->returnType->unwrap());
    finish(">;");
  }

  /* constructor, taking the arguments of the Fiber */
  start("");
  if (!header) {
    middle("bi::" << stateName << "::");
  }
  middle(stateName << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" :");
    in();
    in();
    start("super_type(0, " << (yields.size() + 1) << ')');
    for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
      auto param = *iter;
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
  start("");
  if (header) {
    middle("virtual ");
  }
  middle("std::shared_ptr<bi::FiberState<" << o->returnType->unwrap() <<">> ");
  if (!header) {
    middle("bi::" << stateName << "::");
  }
  middle("clone() const");
  if (header) {
    finish(";\n");
  } else {
    finish(" {");
    in();
    line("return std::make_shared<" << stateName << ">(*this);");
    out();
    line("}\n");
  }

  /* call function */
  start("");
  if (header) {
    middle("virtual ");
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
    genSwitch();
    *this << o->braces->strip();
    genEnd();
    out();
    finish("}\n");
  }

  if (header) {
    line("");
    out();
    line("private:");
    in();

    /* parameters and local variables as class member variables */
    for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
      auto param = dynamic_cast<const Parameter*>(*iter);
      assert(param);
      line(param->type << ' ' << param->name << ';');
    }
    for (auto iter = locals.begin(); iter != locals.end(); ++iter) {
      auto param = dynamic_cast<const LocalVariable*>(*iter);
      assert(param);
      line(
          param->type << ' ' << getName(param->name->str(), param->number) << ';');
    }

    out();
    line("};");
  }

  /* initialisation function */
  start(o->returnType << ' ');
  if (!header) {
    middle("bi::");
  }
  middle(o->name << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return make_fiber<");
    middle(o->returnType->unwrap() << ',' << stateName << ">(");
    for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
      if (iter != o->params->begin()) {
        middle(", ");
      }
      const Parameter* param = dynamic_cast<const Parameter*>(*iter);
      assert(param);
      middle(param->name);
    }
    finish(");");
    out();
    finish("}\n");
  }
}

void bi::CppFiberGenerator::visit(const Return* o) {
  line("goto END;");
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  line("this->value = " << o->single << ';');
  line("this->label = " << label << ';');
  line("return true;");
  line("LABEL" << label << ": ;");
  ++label;
}

void bi::CppFiberGenerator::visit(const Identifier<LocalVariable>* o) {
  middle(getName(o->name->str(), o->target->number));
}

void bi::CppFiberGenerator::visit(const LocalVariable* o) {
  if (inFor || !o->value->isEmpty() || !o->args->isEmpty()
      || !o->brackets->isEmpty()) {
    /* the variable is declared in the fiber state, so there is no need to
     * do anything here unless there is some initialization associated with
     * it */
    inFor = false;
    middle(getName(o->name->str(), o->number));
    genInit(o);
  } else if (o->type->isPointer() && !o->type->isWeak()) {
    /* make sure objects are initialized, not just null pointers */
    auto name = getName(o->name->str(), o->number);
    middle(name << " = bi::make_pointer<" << o->type << ">()");
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
      buf << name << '_';
      result = buf.str();
      counts.insert(std::make_pair(name, 1));
    } else {
      buf << name << '_' << count->second << '_';
      result = buf.str();
      ++count->second;
    }
    names.insert(std::make_pair(number, result));
  } else {
    result = iter->second;
  }
  return result;
}
