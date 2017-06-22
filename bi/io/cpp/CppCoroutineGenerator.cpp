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
  /* collect local variables */
  Gatherer<LocalVariable> locals;
  o->accept(&locals);

  /* supporting class */
  if (header) {
    line("namespace bi {");
    in();
    line("namespace func {");
    out();
    line("class Coroutine_" << o->name << "_ : public Coroutine<" << o->returnType << "> {");
    line("public:");
    in();
  }

  /* constructor, taking the arguments of the coroutine */
  if (!header) {
    start("bi::func::Coroutine_" << o->name << "_::");
  } else {
    start("");
  }
  middle("Coroutine_" << o->name << '_' << o->parens);
  if (header) {
    finish(';');
  } else {
    if (o->parens->tupleSize() > 0) {
      finish(" :");
      in();
      bool before = false;
      for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
        auto param = dynamic_cast<const Parameter*>(*iter);
        assert(param);
        if (before) {
          finish(',');
        }
        before = true;
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

  /* call function */
  if (header) {
    start("virtual ");
  } else {
    start("");
  }
  middle(o->returnType << ' ');
  if (!header) {
    middle("bi::func::Coroutine_" << o->name << "_::");
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
    *this << o->braces;

    line("state = " << state << ';');
    out();
    finish("}\n");
  }

  if (header) {
    line("");
    out();
    line("private:");
    in();

    /* parameters and local variables as class member variables */
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      auto param = dynamic_cast<const Parameter*>(*iter);
      assert(param);
      line(param->type << ' ' << param->name << ';');
    }
    for (auto iter = locals.begin(); iter != locals.end(); ++iter) {
      auto param = dynamic_cast<const LocalVariable*>(*iter);
      assert(param);
      line(param->type << ' ' << param->name << '_' << param->number << "_;");
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
  start("bi::Pointer<bi::Coroutine<" << o->returnType << ">> ");
  if (!header) {
    middle("bi::func::");
  }
  middle(o->name << o->parens);
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return BI_NEW(bi::func::Coroutine_" << o->name << "_)(");
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      if (iter != o->parens->begin()) {
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
  if (header) {
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

void bi::CppCoroutineGenerator::visit(const Identifier<LocalVariable>* o) {
  middle(o->name << '_' << o->target->number << '_');
}

void bi::CppCoroutineGenerator::visit(const LocalVariable* o) {
  if (o->type->isClass() || !o->parens->isEmpty() || !o->value->isEmpty()) {
    middle(o->name << '_' << o->number << '_');
    genInit(o);
    ///@todo This will need to resize arrays, overload operator() for Array?
  }
}
