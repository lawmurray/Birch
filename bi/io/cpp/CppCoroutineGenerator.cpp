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

void bi::CppCoroutineGenerator::visit(const Coroutine* o) {
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
  middle("Coroutine_" << o->name << "_");

  CppParameterGenerator auxParameter(base, level, header);
  auxParameter << o;
  if (header) {
    finish(';');
  } else {
    if (o->parens->tupleSize() > 0) {
      finish(" :");
      in();
      Gatherer<Parameter> gatherer;
      o->parens->accept(&gatherer);
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        const Parameter* param = *iter;
        if (iter != gatherer.begin()) {
          finish(',');
        }
        start(param->name << '_' << param->number << '_' << '(' << param->name << ')');
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
  ++inReturn;
  middle(o->returnType);
  --inReturn;
  middle(' ');
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
    Gatherer<Parameter> gatherer;
    o->accept(&gatherer);
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      const Parameter* param = *iter;
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

  /* return type */
  ++inReturn;
  start("bi::Pointer<bi::Coroutine<" << o->returnType << ">> ");
  --inReturn;

  /* name */
  if (!header) {
    middle("bi::func::");
  }
  middle(o->name);

  /* parameters */
  auxParameter << o;

  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return new (GC_MALLOC(sizeof(bi::func::Coroutine_" << o->name << "_))) bi::func::Coroutine_" << o->name << "_(");
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
  if (o->type->isClass() || o->type->count() > 0) {
    middle(o->name << '_' << o->number << '_');
  }
  if (o->type->isClass()) {
    Identifier<Class>* type = dynamic_cast<Identifier<Class>*>(o->type->strip());
    assert(type);
    middle(" = new (GC_MALLOC(sizeof(bi::type::" << type->name << "))) bi::type::" << type->name << "()");
  }
  if (o->type->count() > 0) {
    ArrayType* type = dynamic_cast<ArrayType*>(o->type->strip());
    assert(type);
    middle("(make_frame(" << type->brackets << "))");
  }
}
