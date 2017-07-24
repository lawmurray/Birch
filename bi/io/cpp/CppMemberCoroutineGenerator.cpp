/**
 * @file
 */
#include "bi/io/cpp/CppMemberCoroutineGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"

bi::CppMemberCoroutineGenerator::CppMemberCoroutineGenerator(
    const Class* type, std::ostream& base, const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    type(type),
    state(0) {
  //
}

void bi::CppMemberCoroutineGenerator::visit(const MemberCoroutine* o) {
  /* gather local variables */
  Gatherer < LocalVariable > locals;
  o->accept(&locals);

  /* gather return statements */
  Gatherer < Return > returns;
  o->braces->accept(&returns);

  /* supporting class */
  if (header) {
    line(
        "class " << o->name << "Coroutine : public Coroutine<" << o->returnType << "> {");
    line("public:");
    in();
  }

  /* constructor, taking the arguments of the coroutine */
  start("");
  if (!header) {
    middle("bi::type::" << type->name << "::" << o->name << "Coroutine::");
  }
  middle(o->name << "Coroutine(Pointer<" << type->name << "> self");
  if (!o->parens->isEmpty()) {
    middle(", " << o->parens->strip());
  }
  middle(')');
  if (header) {
    finish(';');
  } else {
    if (o->parens->tupleSize() > 0) {
      finish(" :");
      in();
      start("self(self)");
      for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
        finish(',');
        auto param = dynamic_cast<const Parameter*>(*iter);
        assert(param);
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

  /* clone function */
  if (!header) {
    start("bi::type::" << type->name << "::");
  } else {
    start("virtual ");
  }
  middle(o->name << "Coroutine* ");
  if (!header) {
    middle("bi::type::" << type->name << "::" << o->name << "Coroutine::");
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
  middle(o->returnType << ' ');
  if (!header) {
    middle("bi::type::" << type->name << "::" << o->name << "Coroutine::");
  }
  middle("operator()()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    if (returns.size() > 0) {
      line("switch (state) {");
      in();
      for (int s = 0; s <= returns.size(); ++s) {
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
    line("Pointer<" << type->name << "> self;");
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
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
  start("bi::Fiber<" << o->returnType << "> ");
  if (!header) {
    middle("bi::type::" << type->name << "::");
  }
  middle(o->name << o->parens);
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return Fiber<" << o->returnType << ">(make_object<");
    middle(o->name << "Coroutine>(pointer_from_this<" << type->name << ">()");
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      middle(", ");
      const Parameter* param = dynamic_cast<const Parameter*>(*iter);
      assert(param);
      middle(param->name);
    }
    finish("));");
    out();
    finish("}\n");
  }
}

void bi::CppMemberCoroutineGenerator::visit(const Return* o) {
  line("state = " << state << "; return " << o->single << ';');
  line("STATE" << state << ": ;");
  ++state;
}

void bi::CppMemberCoroutineGenerator::visit(
    const Identifier<LocalVariable>* o) {
  middle(o->name << o->target->number);
}

void bi::CppMemberCoroutineGenerator::visit(
    const Identifier<MemberVariable>* o) {
  middle("self->" << o->name);
}

void bi::CppMemberCoroutineGenerator::visit(const LocalVariable* o) {
  if (o->type->isClass() || !o->parens->isEmpty() || !o->value->isEmpty()) {
    middle(o->name << o->number);
    genInit(o);
    ///@todo This will need to resize arrays, overload operator() for Array?
  }
}
