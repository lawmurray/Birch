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

  /* gather important objects */
  o->params->accept(&params);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class for state */
  if (header) {
    line("class " << stateName << " : ");
    finish("public FiberState<" << o->returnType->unwrap() << "> {");
    line("public:");
    in();
    line("using super_type = FiberState<" << o->returnType->unwrap() << ">;\n");
    start("SharedCOW<" << type->name);
    genTemplateArgs(type);
    finish("> object;");
    for (auto param : params) {
      line(param->type << ' ' << param->name << ';');
    }
    for (auto local : locals) {
      start(local->type << ' ');
      finish(getName(local->name->str(), local->number) << ';');
    }
  }

  /* constructor */
  start("");
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle(stateName << "(const SharedCOW<");
  middle(type->name);
  genTemplateArgs(type);
  middle(">& object");
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
    start("super_type(0, " << (yields.size() + 1) << ')');
    finish(',');
    start("object(object)");
    for (auto param : params) {
      finish(',');
      start(param->name << '(' << param->name << ')');
    }
    finish(" {");
    out();
    line("//");
    out();
    line("}\n");
  }

  /* self function */
  if (header) {
    line("auto self() {");
    in();
    line("return object.get();");
    out();
    line("}");
  }

  /* clone function */
  if (header) {
    start("virtual ");
  } else {
    start("");
  }
  middle("bi::FiberState<" << o->returnType->unwrap() <<">* ");
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("clone(Memo* memo) const");
  if (header) {
    finish(" override;\n");
  } else {
    finish(" {");
    in();
    line("return bi::clone_object(this, memo);");
    out();
    line("}\n");
  }

  /* destroy function */
  start("");
  if (header) {
    middle("virtual ");
  }
  middle("void ");
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("destroy()");
  if (header) {
    finish(" override;\n");
  } else {
    finish(" {");
    in();
    line("this->size = sizeof(*this);");
    line("this->~" << stateName << "();");
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
    middle("this->self()");
    for (auto param: params) {
      middle(", ");
      middle(param->name);
    }
    finish(");");
    out();
    finish("}\n");
  }
}
