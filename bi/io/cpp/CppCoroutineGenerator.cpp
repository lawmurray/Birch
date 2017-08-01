/**
 * @file
 */
#include "bi/io/cpp/CppCoroutineGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"

bi::CppCoroutineGenerator::CppCoroutineGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    state(0) {
  //
}

void bi::CppCoroutineGenerator::visit(const Coroutine* o) {
  /* gather important objects */
  o->params->accept(&parameters);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class */
  if (header) {
    line("namespace bi {");
    in();
    line("namespace func {");
    out();
    line(
        "class " << o->name << "Coroutine : public Coroutine<" << o->returnType << "> {");
    line("public:");
    in();
  }

  /* constructor, taking the arguments of the coroutine */
  start("");
  if (!header) {
    middle("bi::func::" << o->name << "Coroutine::");
  }
  middle(o->name << "Coroutine" << o->params);
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
    line("nstates = " << (yields.size() + 1) << ';');
    out();
    line("}\n");
  }

  /* clone function */
  if (!header) {
    start("bi::func::");
  } else {
    start("virtual ");
  }
  middle(o->name << "Coroutine* ");
  if (!header) {
    middle("bi::func::" << o->name << "Coroutine::");
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
    middle("bi::func::" << o->name << "Coroutine::");
  }
  middle("run()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genSwitch();
    *this << o->braces;
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
    in();
    line("}");
    out();
    line("}\n");
  }

  /* initialisation function */
  if (header) {
    line("namespace bi {");
    in();
    line("namespace func {");
    out();
  }
  start("bi::Fiber<" << o->returnType << "> ");
  if (!header) {
    middle("bi::func::");
  }
  middle(o->name << o->params);
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return Fiber<" << o->returnType << ">(make_object<");
    middle(o->name << "Coroutine>(");
    for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
      if (iter != o->params->begin()) {
        middle(", ");
      }
      const Parameter* param = dynamic_cast<const Parameter*>(*iter);
      assert(param);
      middle(param->name);
    }
    finish("));");
    out();
    finish("}\n");
  }
  if (header) {
    in();
    line("}");
    out();
    line("}\n");
  }
}

void bi::CppCoroutineGenerator::visit(const Return* o) {
  line("goto END;");
}

void bi::CppCoroutineGenerator::visit(const Yield* o) {
  line("value = " << o->single << ';');
  line("state = " << state << ';');
  line("return true;");
  line("STATE" << state << ": ;");
  ++state;
}

void bi::CppCoroutineGenerator::visit(const Identifier<LocalVariable>* o) {
  middle(o->name << o->target->number);
}

void bi::CppCoroutineGenerator::visit(const LocalVariable* o) {
  if (o->type->isClass() || !o->parens->isEmpty() || !o->value->isEmpty()) {
    middle(o->name << o->number);
    genInit(o);
    ///@todo This will need to resize arrays, overload operator() for Array?
  }
}

void bi::CppCoroutineGenerator::genSwitch() {
  line("switch (state) {");
  in();
  for (int s = 0; s <= yields.size(); ++s) {
    line("case " << s << ": goto STATE" << s << ';');
  }
  line("default: goto END;");
  out();
  line('}');
  line("STATE0:");
  ++state;
}

void bi::CppCoroutineGenerator::genEnd() {
  line("END:");
  line("state = " << (yields.size() + 1) << ';');
  line("return false;");
}
