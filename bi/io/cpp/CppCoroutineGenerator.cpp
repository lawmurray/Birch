/**
 * @file
 */
#include "bi/io/cpp/CppCoroutineGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppCoroutineGenerator::CppCoroutineGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    state(0) {
  //
}

void bi::CppCoroutineGenerator::visit(const FuncParameter* o) {
  /* pre-condition */
  assert(o->isCoroutine());

  if (header) {
    line("namespace bi {");
    in();
    line("namespace func {");
    out();
    line("class " << o->name << " : Coroutine<" << o->type << "> {");
    line("public:");
    in();
  }

  /* constructor, taking the arguments of the coroutine */
  if (!header) {
    start("bi::func::" << o->name << "::");
  } else {
    start("");
  }
  middle(o->name);

  CppParameterGenerator auxParameter(base, level, header);
  auxParameter << o;
  if (header) {
    finish(';');
  } else {
    if (o->parens->tupleSize() > 0) {
      finish(" :");
      in();
      Gatherer<VarParameter> gatherer;
      o->parens->accept(&gatherer);
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        const VarParameter* param = *iter;
        if (iter != gatherer.begin()) {
          finish(',');
        }
        start(param->name << '(' << param->name << ')');
      }
      out();
    }
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }

  /* yield function */
  if (header) {
    start("virtual ");
  } else {
    start("");
  }
  ++inReturn;
  middle(o->type);
  --inReturn;
  middle(' ');
  if (!header) {
    middle("bi::func::" << o->name << "::");
  }
  middle("operator()()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();

    Gatherer<Return> gatherer;
    o->braces->accept(&gatherer);
    if (gatherer.size() > 0) {
      line("switch (state) {");
      in();
      for (int s = 0; s <= gatherer.size(); ++s) {
        line("case " << s << ": goto STATE" << s << ';');
      }
      out();
      line('}');
    }

    line("STATE0:");
    ++state;

    ++inCoroutine;
    *this << o->braces;
    ++inCoroutine;

    line("state = " << state << ';');
    out();
    finish("}\n");
  }

  if (header) {
    line("");
    out();
    line("private:");
    in();

    /* function parameters as class member variables */
    Gatherer<VarParameter> gatherer1;
    o->parens->accept(&gatherer1);
    for (auto iter = gatherer1.begin(); iter != gatherer1.end(); ++iter) {
      const VarParameter* param = *iter;
      line(param->type << ' ' << param->name << ';');
    }
    line("");

    /* local variables as class member variables */
    Gatherer<VarParameter> gatherer2;
    o->braces->accept(&gatherer2);
    for (auto iter = gatherer2.begin(); iter != gatherer2.end(); ++iter) {
      const VarParameter* param = *iter;
      line(param->type << ' ' << param->name << '_' << param->number << "_;");
    }

    out();
    line("};");
    in();
    line("}");
    out();
    line("}\n");
  }
}

void bi::CppCoroutineGenerator::visit(const Return* o) {
  line("state = " << state << "; return " << o->single << ';');
  line("STATE" << state << ": ;");
  ++state;
}
