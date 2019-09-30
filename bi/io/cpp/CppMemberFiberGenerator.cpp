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
  std::string stateName = baseName + "_FiberState_";

  /* gather important objects */
  o->params->accept(&params);
  o->braces->accept(&locals);
  o->braces->accept(&yields);

  /* supporting class for state */
  if (header) {
    start("class " << stateName << " final : ");
    finish("public libbirch::FiberState<" << o->returnType->unwrap() << "> {");
    line("public:");
    in();
    line("using class_type_ = " << stateName << ';');
    line("using super_type_ = libbirch::FiberState<" << o->returnType->unwrap() << ">;\n");
    start("libbirch::Shared<bi::type::" << type->name);
    genTemplateArgs(type);
    finish("> self;");
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
  middle(stateName << "(const libbirch::Shared<");
  middle(type->name);
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
    start("super_type_(0, " << (yields.size() + 1) << ')');
    finish(',');
    start("self(self)");
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

  /* deep copy constructor */
  start("");
  if (!header) {
    middle("bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle(stateName << "(const " << stateName << "& o_, int)");
  if (header) {
    finish(';');
  } else {
    finish(" :");
    in();
    in();
    start("super_type_(o_, 0)");
    finish(',');
    start("self(o_.self, 0)");
    for (auto o : params) {
      finish(',');
      start(o->name << "(libbirch::clone(o_." << o->name << "))");
    }
    for (auto o : locals) {
      auto name = getName(o->name->str(), o->number);
      finish(',');
      start(name << "(libbirch::clone(o_." << name << "))");
    }
    finish(" {");
    out();
    line("//");
    out();
    line("}\n");
  }

  /* destructor, assignment operator */
  if (header) {
    line("virtual ~" << stateName << "() = default;");
    line(stateName << "(const " << stateName << "&) = delete;");
    line(stateName << "& operator=(const " << stateName << "&) = delete;");
  }

  if (header) {
    out();
    line("public:");
    in();
  }

  /* standard functions */
  if (header) {
    line("libbirch_create_function_");
    line("libbirch_emplace_function_");
    line("libbirch_clone_function_");
    line("libbirch_destroy_function_");
  }

  /* freeze function */
  line("#if ENABLE_LAZY_DEEP_CLONE");
  if (header) {
    start("virtual void ");
  } else {
    start("void bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("doFreeze_()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    line("super_type_::doFreeze_();");
    line("libbirch::freeze(self);");
    line("libbirch::freeze(value_);");
    for (auto param : params) {
      line("libbirch::freeze(" << param->name << ");");
    }
    for (auto local : locals) {
      line("libbirch::freeze(" << getName(local->name->str(), local->number) << ");");
    }
    out();
    line("}\n");
  }

  /* thaw function */
  if (header) {
    start("virtual void ");
  } else {
    start("void bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("doThaw_(libbirch::LazyContext* context)");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    line("super_type_::doThaw_(context);");
    line("libbirch::thaw(self, context);");
    line("libbirch::thaw(value_, context);");
    for (auto param : params) {
      line("libbirch::thaw(" << param->name << ", context);");
    }
    for (auto local : locals) {
      line("libbirch::thaw(" << getName(local->name->str(), local->number) << ", context);");
    }
    out();
    line("}\n");
  }

  /* finish function */
  if (header) {
    start("virtual void ");
  } else {
    start("void bi::type::" << type->name);
    genTemplateArgs(type);
    middle("::" << stateName << "::");
  }
  middle("doFinish_()");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    line("super_type_::doFinish_();");
    line("libbirch::finish(self);");
    line("libbirch::finish(value_);");
    for (auto param : params) {
      line("libbirch::finish(" << param->name << ");");
    }
    for (auto local : locals) {
      line("libbirch::finish(" << getName(local->name->str(), local->number) << ");");
    }
    out();
    line("}");
  }
  line("#endif\n");

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
    line("libbirch_swap_context_");
    line("libbirch_declare_local_");
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
    line("libbirch_swap_context_");
    line("libbirch_declare_self_");
    start("return libbirch::make_fiber<" << stateName << ">(");
    middle("self");
    for (auto param: params) {
      middle(", ");
      middle(param->name);
    }
    finish(");");
    out();
    finish("}\n");
  }
}

void bi::CppMemberFiberGenerator::visit(const This* o) {
  middle("local->self");
}

void bi::CppMemberFiberGenerator::visit(const Super* o) {
  middle("local->self");
}
