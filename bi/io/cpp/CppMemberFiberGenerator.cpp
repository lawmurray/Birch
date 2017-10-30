/**
 * @file
 */
#include "bi/io/cpp/CppMemberFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

#include <sstream>

bi::CppMemberFiberGenerator::CppMemberFiberGenerator(const Class* type,
    std::ostream& base, const int level, const bool header) :
    CppFiberGenerator(base, level, header),
    type(type),
    inMember(0) {
  //
}

void bi::CppMemberFiberGenerator::visit(const MemberFiber* o) {
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
    genTemplateParams(type);
    middle("bi::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle(stateName << "(const Pointer<" << type->name);
  genTemplateArgs(type);
  middle(">& self");
  if (!o->params->isEmpty()) {
    middle(", " << o->params);
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
    line("this->nlabels = " << (yields.size() + 1) << ';');
    out();
    line("}\n");
  }

  /* clone function */
  if (!header) {
    genTemplateParams(type);
    if (type->isGeneric()) {
      start("typename ");
    } else {
      start("");
    }
    middle("bi::" << type->name);
    genTemplateArgs(type);
    middle("::");
  } else {
    start("virtual ");
  }
  middle(stateName << "* ");
  if (!header) {
    middle("bi::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
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
  if (header) {
    start("virtual ");
  } else {
    genTemplateParams(type);
    start("");
  }
  middle("bool ");
  if (!header) {
    middle("bi::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
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
    start("Pointer<" << type->name);
    genTemplateArgs(type);
    finish("> self;");
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
  if (!header) {
    genTemplateParams(type);
  }
  start(o->returnType << ' ');
  if (!header) {
    middle("bi::" << type->name);
    genTemplateArgs(type);
    middle("::");
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
    middle('<' << o->returnType->unwrap() << ',' << stateName << '>');
    middle("(pointer_from_this<this_type>()");
    for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
      middle(", ");
      middle((*iter)->name);
    }
    finish(");");
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
  const This* leftThis = dynamic_cast<const This*>(o->left);
  const Super* leftSuper = dynamic_cast<const Super*>(o->left);
  if (leftThis) {
    middle("self->");
  } else if (leftSuper) {
    middle("self->super_type::");
  } else {
    middle(o->left << "->");
  }
  ++inMember;
  middle(o->right);
  --inMember;
}

void bi::CppMemberFiberGenerator::visit(const This* o) {
  middle("self");
}

void bi::CppMemberFiberGenerator::visit(const Super* o) {
  middle("self");
}
