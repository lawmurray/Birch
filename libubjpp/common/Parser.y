%include {
  #include <libubjpp/common/ParserState.h>
  #include <assert.h>
}

%stack_size 0
%extra_argument { ParserState* state }
%syntax_error   { error(state); }

root ::= object.
root ::= array.

left_brace ::= LEFT_BRACE.  { object(state); }

object ::= left_brace RIGHT_BRACE.
object ::= left_brace members RIGHT_BRACE.

members ::= key COLON value.                { member(state); }
members ::= members COMMA key COLON value.  { member(state); }

left_bracket ::= LEFT_BRACKET.  { array(state); }

array ::= left_bracket RIGHT_BRACKET.
array ::= left_bracket elements RIGHT_BRACKET.

elements ::= value.                 { element(state); }
elements ::= elements COMMA value.  { element(state); }

key ::= STRING.  { push(state); }

value ::= object.
value ::= array.
value ::= STRING.  { push(state); }
value ::= INT8.    { push(state); }
value ::= UINT8.   { push(state); }
value ::= INT16.   { push(state); }
value ::= INT32.   { push(state); }
value ::= INT64.   { push(state); }
value ::= FLOAT.   { push(state); }
value ::= DOUBLE.  { push(state); }
value ::= BOOL.    { push(state); }
value ::= NIL.     { push(state); }
value ::= NO_OP.   { push(state); }
