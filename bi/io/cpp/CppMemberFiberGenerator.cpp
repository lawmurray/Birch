/**
 * @file
 */
#include "bi/io/cpp/CppMemberFiberGenerator.hpp"

bi::CppMemberFiberGenerator::CppMemberFiberGenerator(const Class* type,
    std::ostream& base, const int level, const bool header) :
    CppFiberGenerator(base, level, header),
    type(type),
    state(0),
    inMember(0) {
  //
}

void bi::CppMemberFiberGenerator::visit(const MemberFiber* o) {
  /* gather important objects */
  o->params->accept(&parameters);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class */
  if (header) {
    line(
        "class " << o->name << "FiberState : public FiberState<" << o->returnType << "> {");
    line("public:");
    in();
  }

  /* constructor, taking the arguments of the Fiber */
  start("");
  if (!header) {
    middle("bi::type::" << type->name << "::" << o->name << "FiberState::");
  }
  middle(o->name << "FiberState(const Pointer<" << type->name << ">& self");
  if (!o->params->strip()->isEmpty()) {
    middle(", " << o->params->strip());
  }
  middle(')');
  if (header) {
    finish(';');
  } else {
    finish(" :");
    in();
    in();
    start("self(self)");
    for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
      finish(',');
      start((*iter)->name << '(' << (*iter)->name << ')');
    }
    out();
    out();
    finish(" {");
    in();
    line("nstates = " << (yields.size() + 1) << ';');
    out();
    line("}\n");
  }

  /* clone function */
  if (!header) {
    start("bi::type::" << type->name << "::");
  } else {
    start("virtual ");
  }
  middle(o->name << "FiberState* ");
  if (!header) {
    middle("bi::type::" << type->name << "::" << o->name << "FiberState::");
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
    middle("bi::type::" << type->name << "::" << o->name << "FiberState::");
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
    line("Pointer<" << type->name << "> self;");
    for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
      line((*iter)->type << ' ' << (*iter)->name << ';');
    }
    for (auto iter = locals.begin(); iter != locals.end(); ++iter) {
      line((*iter)->type << ' ' << (*iter)->name << (*iter)->number << ';');
    }

    out();
    line("};");
  }

  /* initialisation function */
  start("bi::Fiber<" << o->returnType << "> ");
  if (!header) {
    middle("bi::type::" << type->name << "::");
  }
  middle(o->name << o->params);
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return Fiber<" << o->returnType << ">(make_object<");
    middle(
        o->name << "FiberState>(pointer_from_this<" << type->name << ">()");
    for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
      middle(", ");
      middle((*iter)->name);
    }
    finish("));");
    out();
    finish("}\n");
  }
}

void bi::CppMemberFiberGenerator::visit(const Identifier<MemberVariable>* o) {
  if (!inMember) {
    middle("self->");
  }
  middle(o->name);
}

void bi::CppMemberFiberGenerator::visit(
    const Identifier<MemberParameter>* o) {
  if (!inMember) {
    middle("self->");
  }
  middle(o->name);
}

void bi::CppMemberFiberGenerator::visit(
    const OverloadedIdentifier<MemberFunction>* o) {
  if (!inMember) {
    middle("self->");
  }
  middle(o->name);
}

void bi::CppMemberFiberGenerator::visit(
    const OverloadedIdentifier<MemberFiber>* o) {
  if (!inMember) {
    middle("self->");
  }
  middle(o->name);
}

void bi::CppMemberFiberGenerator::visit(const Member* o) {
  ++inMember;
  const Super* leftSuper = dynamic_cast<const Super*>(o->left);
  if (leftSuper) {
    // tidier this way
    middle("super_type::");
  } else {
    middle(o->left << "->");
  }
  middle(o->right);
  --inMember;
}

void bi::CppMemberFiberGenerator::visit(const This* o) {
  middle("self");
}
