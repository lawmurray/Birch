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
    line("using class_type = " << stateName << ';');
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

  /* self-reference function */
  if (header) {
    out();
    line("private:");
    in();
    line("auto self() {");
    in();
    line("return object;");
    out();
    line("}\n");
    line("auto local() {");
    in();
    line("return SharedCOW<class_type>(this, context.get());");
    out();
    line("}\n");
  }

  if (header) {
    out();
    line("protected:");
    in();
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

  /* copy constructor, destructor, assignment operator */
  if (header) {
    line(stateName << "(const " << stateName << "&) = default;");
    line("virtual ~" << stateName << "() = default;");
    line(stateName << "& operator=(const " << stateName << "&) = default;");
  }

  if (header) {
    out();
    line("public:");
    in();
  }

  /* standard functions */
  if (header) {
    line("STANDARD_CREATE_FUNCTION");
    line("STANDARD_EMPLACE_FUNCTION");
    line("STANDARD_CLONE_FUNCTION");
    line("STANDARD_DESTROY_FUNCTION");
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
    middle("self()");
    for (auto param: params) {
      middle(", ");
      middle(param->name);
    }
    finish(");");
    out();
    finish("}\n");
  }
}
