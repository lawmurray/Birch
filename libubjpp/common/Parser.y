%include {
  #include <libubjpp/common/ParserState.h>
  #include <assert.h>
}
%extra_argument { ParserState* state }
%syntax_error   { error(state); }

root ::= object.

object ::= LEFT_BRACE RIGHT_BRACE.
object ::= LEFT_BRACE members RIGHT_BRACE.

members ::= STRING COLON value.                { member(state); }
members ::= members COMMA STRING COLON value.  { member(state); }

array ::= LEFT_BRACKET RIGHT_BRACKET.
array ::= LEFT_BRACKET elements RIGHT_BRACKET.

elements ::= value.                 { element(state); }
elements ::= elements COMMA value.  { element(state); }

value ::= object.
value ::= array.
value ::= STRING.
value ::= INT8.
value ::= UINT8.
value ::= INT16.
value ::= INT32.
value ::= INT64.
value ::= FLOAT.
value ::= DOUBLE.
value ::= BOOL.
value ::= NIL.
value ::= NO_OP.
