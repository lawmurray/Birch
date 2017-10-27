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
    label(0) {
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
    finish("public FiberState<" << o->returnType->unwrap() << "> {");
    line("public:");
    in();
  }

  /* constructor, taking the arguments of the Fiber */
  start("");
  if (!header) {
    middle("bi::func::" << stateName << "::");
  }
  middle(stateName << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    if (o->params->tupleSize() > 0) {
      finish(" :");
      in();
      for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
        if (iter != parameters.begin()) {
          finish(',');
        }
        start((*iter)->name << '(' << (*iter)->name << ')');
      }
      out();
    }
    finish(" {");
    in();
    line("this->nlabels = " << (yields.size() + 1) << ';');
    out();
    line("}\n");
  }

  /* clone function */
  if (!header) {
    start("bi::func::");
  } else {
    start("virtual ");
  }
  middle(stateName << "* ");
  if (!header) {
    middle("bi::func::" << stateName << "::");
  }
  middle("clone()");
  if (header) {
    finish(";\n");
  } else {
    finish(" {");
    in();
    line("return copy_object(this);");
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
    middle("bi::func::" << stateName << "::");
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
      line(param->type << ' ' << param->name << param->number << ';');
    }

    out();
    line("};");
  }

  /* initialisation function */
  start(o->returnType << ' ');
  if (!header) {
    middle("bi::func::");
  }
  middle(o->name << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return ");
    if (o->isClosed()) {
      middle("make_closed_fiber");
    } else {
      middle("make_fiber");
    }
    middle('<' << o->returnType->unwrap() << ',' << stateName << ">(");
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
  // there may be local variables in the fiber body that have the same name,
  // distinguished only by scope; as these scopes are now collapsed, local
  // variable names are suffixed with a unique number to distinguish them
  middle(o->name << o->target->number);
}

void bi::CppFiberGenerator::visit(const LocalVariable* o) {
  // see above for use of number here
  middle(o->name << o->number);
  genInit(o);
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
