/**
 * @file
 */
#include "bi/io/cpp/CppMemberFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/primitive/encode.hpp"

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
  std::string baseName = internalise(o->name->str()) + encode32(base.str());
  std::string stateName = baseName + "_FiberState";
  std::string localName = baseName + "_FiberLocal";
  std::string argName = baseName + "_FiberArg";

  /* gather important objects */
  o->params->accept(&params);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class for arguments */
  if (header) {
    line("class " << argName << " {");
    in();
    line("public:");
    for (auto param : params) {
      line(param->type << ' ' << param->name << ';');
    }
    out();
    line("};\n");
  }

  /* supporting class for local variables */
  if (header) {
    line("class " << localName << " {");
    in();
    line("public:");
    for (auto local : locals) {
      start(local->type << ' ');
      finish(getName(local->name->str(), local->number) << ';');
    }
    out();
    line("};\n");
  }

  /* supporting class for state */
  if (header) {
    line("class " << stateName << " : ");
    in();
    in();
    start("public MemberFiberState<" << o->returnType->unwrap() << ',');
    middle(type->name);
    genTemplateArgs(type);
    middle(',');
    middle(argName << ',');
    middle(localName << '>');
    finish(" {");
    out();
    out();
    line("public:");
    in();
    start("using state_type = MemberFiberState<");
    middle(o->returnType->unwrap() << ',');
    middle(type->name);
    genTemplateArgs(type);
    middle(',');
    middle(argName << ',');
    middle(localName << '>');
    finish(';');
  }

  /* constructor */
  line("template<class... Args>");
  start("");
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle(stateName << "(const SharedPointer<");
  middle(type->name);
  genTemplateArgs(type);
  middle(">& object, Args... args)");
  if (header) {
    finish(';');
  } else {
    finish(" :");
    in();
    in();
    line("state_type(0, " << (yields.size() + 1) << ", object, args...) {");
    out();
    line("//");
    out();
    line("}\n");
  }

  /* clone function */
  if (header) {
    start("virtual ");
  } else {
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
    line("return std::allocate_shared<" << stateName << ">(PowerPoolAllocator<" << stateName << ">(), *this);");
    out();
    line("}\n");
  }

  /* query function */
  if (header) {
    start("virtual ");
  } else {
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
    genTraceFunction(o->name->str(), o->loc);
    genSwitch();
    *this << o->braces->strip();
    genEnd();
    out();
    finish("}\n");
  }
  if (header) {
    line("};\n");
  }

  /* initialisation function */
  if (header) {
    start("virtual ");
  } else {
    start("");
  }
  middle(o->returnType << ' ');
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::");
  }
  middle(o->name << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    start("return make_fiber<" << stateName << ">(");
    middle("this->shared_self()");
    for (auto param: params) {
      middle(", ");
      middle(param->name);
    }
    finish(");");
    out();
    finish("}\n");
  }
}
