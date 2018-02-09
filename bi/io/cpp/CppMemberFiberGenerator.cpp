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
    type(type) {
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
    middle("public MemberFiberState<" << o->returnType->unwrap() << ',');
    middle(type->name);
    genTemplateArgs(type);
    finish("> {");
    line("public:");
    in();
    start("using fiber_super_type = MemberFiberState<");
    middle(o->returnType->unwrap() << ',');
    middle(type->name);
    genTemplateArgs(type);
    finish(">;");
  }

  /* constructor, taking the arguments of the Fiber */
  start("");
  if (!header) {
    genTemplateParams(type);
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle(stateName << "(const SharedPointer<");
  if (o->isReadOnly()) {
    middle("const ");
  }
  middle(type->name);
  genTemplateArgs(type);
  middle(">& o");
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
    line("fiber_super_type(o, 0, " << (yields.size() + 1) << ")");
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
  if (header) {
    start("virtual ");
  } else {
    genTemplateParams(type);
    start("");
  }
  middle("std::shared_ptr<bi::FiberState<" << o->returnType->unwrap() <<">> ");
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
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

  /* query function */
  if (header) {
    start("virtual ");
  } else {
    genTemplateParams(type);
    start("");
  }
  middle("bool ");
  if (!header) {
    middle("bi::type::" << type->name);
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
    for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
      line((*iter)->type << ' ' << (*iter)->name << ';');
    }
    for (auto iter = locals.begin(); iter != locals.end(); ++iter) {
      line((*iter)->type << ' ' << getName((*iter)->name->str(), (*iter)->number) << ';');
    }

    out();
    line("};");
  }

  /* call function */
  if (header) {
    start("virtual ");
  } else {
    genTemplateParams(type);
    start("");
  }
  middle(o->returnType << ' ');
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::");
  }
  middle(o->name << '(' << o->params << ')');
  if (o->isReadOnly()) {
    middle(" const");
  }
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return make_member_fiber<");
    middle(o->returnType->unwrap() << ',' << stateName << '>');
    middle("(shared_from_this<this_type>()");
    for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
      middle(", ");
      middle((*iter)->name);
    }
    finish(");");
    out();
    finish("}\n");
  }
}
