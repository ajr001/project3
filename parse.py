"""
EXPR          ::= LOGICAL_OR
LOGICAL_OR    ::= LOGICAL_AND { '||' LOGICAL_AND }
LOGICAL_AND   ::= COMPARISON { '&&' COMPARISON }
COMPARISON    ::= CONCAT [ (EQ | NEQ | LT | LE | GT | GE) CONCAT ]
CONCAT        ::= ADDITIVE { AMP ADDITIVE }
ADDITIVE      ::= MULTIPLICATIVE { (PLUS | MINUS) MULTIPLICATIVE }
MULTIPLICATIVE::= UNARY { (STAR | SLASH) UNARY }
UNARY         ::= [ (PLUS | MINUS | BANG) ] PRIMARY
PRIMARY       ::= NUMBER | STRING | BOOLEAN | RANGE_REF | CELL_REF | 
                 | FUNCTION_CALL | LPAREN EXPR RPAREN | PRIMARY (LBRACK EXPR RBRACK | DOT IDENTIFIER)*
FUNCTION_CALL ::= PRIMARY LPAREN [ ARG_LIST ] RPAREN
ARG_LIST      ::= EXPR { COMMA EXPR }
"""

def dbg_print(*args):
    pass 

from tokens import (
    Int, String, Bool, Null, Identifier as TokenIdentifier, Dot, Comma, 
    Plus, Minus, Star, Slash, Amp, 
    EqEq, Neq, Lt, Le, Gt, Ge, And, Or, Bang,
    LParen, RParen, LBrack, RBrack, 
    matcher, get_tokens, Token, NO_MATCH
)

INDENT = "    "  # 4 spaces for tree rendering

def _render_child(node, indent):
    """Helper: render child node, using its to_tree if present."""
    if hasattr(node, "to_tree"):
        return node.to_tree(indent)
    return f"{INDENT*indent}{node}"

class ASTNode:
    """Base class for all AST nodes."""
    def _tree_label(self):
        return self.__class__.__name__
        
    def _tree_children(self):
        return []
        
    def __str__(self):
        return str(self.value) if hasattr(self, 'value') else self.__class__.__name__
        
    def to_tree(self, level=0):
        """Convert the AST to a tree string representation."""
        indent = '  ' * level
        label = self._tree_label()
        result = [f"{indent}{label}"]
        
        for child in self._tree_children():
            if child is not None:
                if hasattr(child, 'to_tree'):
                    result.append(child.to_tree(level + 1))
                else:
                    result.append(f"{indent}  {child}")
                    
        return '\n'.join(result)

class UnaryOp(ASTNode):
    """AST node for unary operators (+, -, !)."""
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand
        
    def __str__(self):
        return f"{self.op}{self.operand}"
        
    def _tree_label(self):
        return f"UNARY_{self.op.upper()}"
        
    def _tree_children(self):
        return [self.operand]

class ParenExp(ASTNode):
    """Parenthesized expression node."""
    def __init__(self, expr):
        self.expr = expr
    def __str__(self):
        return f"({self.expr})"
    __repr__ = __str__
    def _tree_label(self):
        return "()"
    def _tree_children(self):
        return [self.expr]

class Identifier(ASTNode):
    """AST node for identifiers."""
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return self.name
        
    def _tree_label(self):
        return f'ID:{self.name}'
        
    @classmethod
    def from_token(cls, token):
        """Create an Identifier AST node from a token."""
        return cls(token.value)

class Operator(ASTNode):
    """Base class for all operators"""
    def __init__(self, op, left, right=None):
        self.op = op
        self.left = left
        self.right = right
        
    def _needs_parens(self, node):
        """Check if a node needs parentheses in string representation."""
        if not isinstance(node, Operator):
            return False
        return self._precedence() > node._precedence()
        
    def _precedence(self):
        """Return operator precedence (higher number = higher precedence)."""
        # Unary operators (self.right is None) have highest precedence
        if self.right is None:
            return 5
        # Binary operators precedence table (higher number -> tighter binding)
        if self.op in ('&&', '||'):
            return 1  # Lowest precedence
        if self.op in ('==', '!=', '<', '>', '<=', '>='):
            return 2
        if self.op in ('+', '-', '&'):
            return 3
        if self.op in ('*', '/', '%'):
            return 4
        # Fallback for any other operator
        return 0
        
    def _format_operand(self, node):
        """Format an operand with parentheses if needed."""
        if self._needs_parens(node):
            return f"({node})"
        return str(node)
        
    def __str__(self):
        if self.right is None:
            # Unary operator
            return f"{self.op}{self._format_operand(self.left)}"
            
        # Binary operator
        left_str = self._format_operand(self.left)
        right_str = self._format_operand(self.right)
        
        # Always add spaces around binary operators for readability
        return f"{left_str} {self.op} {right_str}"
    __repr__ = __str__

class BinaryOperator(Operator):
    """Base for all binary operators (infix)"""
    def _tree_children(self):
        return [self.left, self.right]

class UnaryOperator(Operator):
    """Base for unary operators (prefix)"""
    def _tree_children(self):
        return [self.right]  # left is None for unary

class ArithmeticOp(BinaryOperator):
    """Arithmetic operators: +, -, *, /"""
    def _tree_label(self):
        return f"ARITH_{self.op}"

class ComparisonOp(BinaryOperator):
    """Comparison operators: ==, !=, <, >, <=, >="""
    def _tree_label(self):
        return f"CMP_{self.op}"

class LogicalOp(BinaryOperator):
    """Logical operators: &&, ||"""
    def _tree_label(self):
        return f"LOGIC_{self.op}"

class UnaryOp(UnaryOperator):
    """Unary operators: +, -, !"""
    def _tree_label(self):
        return f"UNARY_{self.op}"

class Member(ASTNode):
    """Member access AST node (e.g. a.b)."""
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
    def __str__(self):
        return f"{self.obj}.{self.attr}"
    __repr__ = __str__

    def _tree_label(self):
        return "."
    def _tree_children(self):
        return [self.obj, self.attr]

def _parse_arglist(ts):
    """Parse argument list inside parentheses, returning list of args or NO_MATCH."""
    start = ts.position
    if ts.match(LParen) is NO_MATCH:
        return NO_MATCH
    
    args = []
    # Check for empty call
    if ts.match(RParen) is NO_MATCH:
        # parse first argument
        arg = Expr.match()(ts)
        if arg is NO_MATCH:
            ts.position = start
            return NO_MATCH
        args.append(arg)
        
        while ts.match(Comma) is not NO_MATCH:
            arg = Expr.match()(ts)
            if arg is NO_MATCH:
                ts.position = start
                return NO_MATCH
            args.append(arg)
        
        if ts.match(RParen) is NO_MATCH:
            ts.position = start
            return NO_MATCH
    
    return args

class Index(ASTNode):
    """Array indexing AST node: array[index]"""
    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __str__(self):
        return f"{self.array}[{self.index}]"
    __repr__ = __str__

    def _tree_label(self):
        return "[]"
    def _tree_children(self):
        return [self.array, self.index]

class Call(ASTNode):
    """Function call AST node: primary(arg1, arg2, ...)"""
    def __init__(self, func, args):
        self.func = func  # any primary expression
        self.args = args  # list of Expr

    @classmethod
    def from_primary(cls, primary):
        """Create a call node from a primary expression."""
        def _match(ts):
            args = _parse_arglist(ts)
            if args is NO_MATCH:
                return NO_MATCH
            return cls(primary, args)
        return matcher(_match)

    def __str__(self):
        return f"{self.func}({','.join(str(a) for a in self.args)})"
    __repr__ = __str__

    def _tree_label(self):
        return "CALL"
    def _tree_children(self):
        return [self.func] + self.args


class TokenStream:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0
        dbg_print(f"\n=== TokenStream created with {len(tokens)} tokens ===")
        for i, tok in enumerate(tokens):
            dbg_print(f"  {i}: {tok.__class__.__name__}({tok})")

    def peek(self):
        ret = None
        if self.position < len(self.tokens):
            ret = self.tokens[self.position]
            dbg_print(f"  peek() -> {ret.__class__.__name__}({ret})")
        else:
            dbg_print("  peek() -> EOF")
        return ret

    def consume(self):
        ret = None
        if self.position < len(self.tokens):
            ret = self.tokens[self.position]
            dbg_print(f"  consume() -> {ret.__class__.__name__}({ret})")
            self.position += 1
        else:
            dbg_print("  consume() -> EOF")
        return ret

    def match(self, cls):
        tok = self.peek()
        if isinstance(tok, cls):
            dbg_print(f"  match({cls.__name__}) -> MATCHED {tok} (type: {type(tok).__name__})")
            return self.consume()
        else:
            dbg_print(f"  match({cls.__name__}) -> NO_MATCH (got {tok.__class__.__name__ if tok is not None else 'EOF'}, expected {cls.__name__})")
            return NO_MATCH

def _token_to_ast(token):
    """Convert a token to its corresponding AST node."""
    dbg_print(f"\n_token_to_ast() - Converting token: {token} (type: {type(token).__name__})")
    
    if isinstance(token, TokenIdentifier):
        result = Identifier(token.value)
        dbg_print(f"  Converted Identifier token to AST node: {result}")
        return result
    elif isinstance(token, (Int, String, Bool, Null)):
        # These tokens are already in the correct format for the AST
        dbg_print(f"  Token is already in AST format: {token}")
        return token
    
    dbg_print(f"  No conversion needed for token: {token}")
    return token  # For any other token types, just return as-is

class Primary:
    @classmethod
    def match(cls):
        def _match(ts):
            start_pos = ts.position
            dbg_print(f"\nPrimary.match() - Starting at position {start_pos}")

            # Try literals and identifiers first
            ret = NO_MATCH
            for token_cls in (Int, String, Bool, Null, TokenIdentifier):
                dbg_print(f"  Trying to match {token_cls.__name__} at position {ts.position}")
                tok = ts.match(token_cls)
                if tok is not NO_MATCH:
                    dbg_print(f"  Matched {token_cls.__name__} token: {tok}")
                    ret = _token_to_ast(tok)
                    dbg_print(f"  Converted to AST node: {ret}")
                    break
                else:
                    dbg_print(f"  No {token_cls.__name__} match")

            if ret is NO_MATCH:
                dbg_print("  No literal or identifier matched, trying parenthesized expression")
                # Parenthesized expression
                if ts.match(LParen) is not NO_MATCH:
                    dbg_print("    Found '(', parsing expression")
                    expr = Expr.match()(ts)
                    if expr is not NO_MATCH and ts.match(RParen) is not NO_MATCH:
                        dbg_print("    Found matching ')', creating parenthesized expression")
                        ret = ParenExp(expr)
                    else:
                        dbg_print("    No matching ')' found, rewinding")
                        ts.position = start_pos
                        return NO_MATCH
                else:
                    dbg_print("    No parenthesized expression found")
                    return NO_MATCH

            # Postfix operators (member access, function calls, array indexing)
            while True:
                start = ts.position
                dbg_print(f"\n  Checking for postfix operators at position {start}")

                # Member access: .identifier
                if ts.match(Dot) is not NO_MATCH:
                    dbg_print("    Found '.', looking for identifier")
                    attr = ts.match(TokenIdentifier)
                    if attr is not NO_MATCH:
                        dbg_print(f"    Found member access: .{attr}")
                        ret = Member(ret, attr)
                        continue
                    else:
                        dbg_print("    No identifier after '.', rewinding")
                        ts.position = start

                # Function call: (arg1, arg2, ...)
                dbg_print("    Checking for function call")
                args = _parse_arglist(ts)
                if args is not NO_MATCH:
                    dbg_print(f"    Found function call with {len(args)} arguments")
                    ret = Call(ret, args)
                    continue
                else:
                    dbg_print("    No function call found")

                # Array indexing: [expr]
                if ts.match(LBrack) is not NO_MATCH:
                    dbg_print("    Found '[' for array indexing")
                    index = Expr.match()(ts)
                    if index is not NO_MATCH and ts.match(RBrack) is not NO_MATCH:
                        dbg_print(f"    Created array index: {ret}[{index}]")
                        ret = Index(ret, index)
                        continue
                    else:
                        dbg_print("    Invalid array index, rewinding")
                        ts.position = start

                # No more postfix operators
                dbg_print("    No more postfix operators")
                break

            dbg_print(f"  Final primary expression: {ret}")
            return ret
        return matcher(_match)

class Unary:
    @classmethod
    def match(cls):
        def _match(ts):
            start_pos = ts.position
            dbg_print(f"\nUnary.match() - Starting at position {start_pos}")
            
            # First try to match a unary operator
            if ts.match(Plus) is not NO_MATCH:
                op = '+'
                dbg_print(f"  Found unary + at position {start_pos}")
            elif ts.match(Minus) is not NO_MATCH:
                op = '-'
                dbg_print(f"  Found unary - at position {start_pos}")
            elif ts.match(Bang) is not NO_MATCH:
                op = '!'
                dbg_print(f"  Found unary ! at position {start_pos}")
            else:
                # No unary operator found, try primary expression
                dbg_print(f"  No unary operator at position {start_pos}, trying primary")
                result = Primary.match()(ts)
                dbg_print(f"  Primary match at {start_pos}: {result}")
                return result
            
            # We found a unary operator, now parse the operand
            dbg_print(f"  Found unary {op}, now parsing operand from position {ts.position}")
            current_token = ts.peek() if hasattr(ts, 'peek') else 'N/A'
            dbg_print(f"  Current token: {current_token} (type: {type(current_token).__name__ if current_token != 'N/A' else 'N/A'})")
            
            # Recursively parse unary expressions (to handle multiple unary operators like !!true)
            dbg_print("  Recursively parsing unary expression...")
            operand = cls.match()(ts)
            
            if operand is NO_MATCH:
                dbg_print(f"  Failed to parse operand for {op} at position {start_pos}")
                dbg_print(f"  Current position: {ts.position}, tokens: {[str(t) for t in ts.tokens]}")
                return NO_MATCH
                
            # Create the appropriate unary operator node
            result = UnaryOp(op, operand)
            dbg_print(f"  Created unary operator {op} with operand {operand}")
            dbg_print(f"  Unary expression complete: {result}")
            return result
                
        return matcher(_match)

class Multiplicative:
    @classmethod
    def match(cls):
        def _match(ts):
            dbg_print("\nMultiplicative.match() - Starting")
            
            dbg_print("  Parsing left-hand side (Unary)")
            left = Unary.match()(ts)
            if left is NO_MATCH:
                dbg_print("  No left-hand side found")
                return NO_MATCH
            dbg_print(f"  Left-hand side: {left}")
            
            while True:
                if ts.match(Star) is not NO_MATCH:
                    op = '*'
                    dbg_print("  Found '*' operator")
                elif ts.match(Slash) is not NO_MATCH:
                    op = '/'
                    dbg_print("  Found '/' operator")
                else:
                    dbg_print("  No more multiplicative operators")
                    break
                
                dbg_print("  Parsing right-hand side (Unary)")
                right = Unary.match()(ts)
                if right is NO_MATCH:
                    dbg_print("  No right-hand side found, rewinding")
                    ts.position -= 1  # Rewind the operator
                    break
                    
                dbg_print(f"  Right-hand side: {right}")
                left = ArithmeticOp(op, left, right)
                dbg_print(f"  Created arithmetic operation: {left}")
            
            dbg_print(f"  Final Multiplicative: {left}")
            return left
            
        return matcher(_match)

class Additive:
    @classmethod
    def match(cls):
        def _match(ts):
            dbg_print("\nAdditive.match() - Starting")
            
            dbg_print("  Parsing left-hand side (Multiplicative)")
            left = Multiplicative.match()(ts)
            if left is NO_MATCH:
                dbg_print("  No left-hand side found")
                return NO_MATCH
            dbg_print(f"  Left-hand side: {left}")
            
            while True:
                if ts.match(Plus) is not NO_MATCH:
                    op = '+'
                    dbg_print("  Found '+' operator")
                elif ts.match(Minus) is not NO_MATCH:
                    op = '-'
                    dbg_print("  Found '-' operator")
                else:
                    dbg_print("  No more additive operators")
                    break
                
                dbg_print("  Parsing right-hand side (Multiplicative)")
                right = Multiplicative.match()(ts)
                if right is NO_MATCH:
                    dbg_print("  No right-hand side found, rewinding")
                    ts.position -= 1  # Rewind the operator
                    break
                    
                dbg_print(f"  Right-hand side: {right}")
                left = ArithmeticOp(op, left, right)
                dbg_print(f"  Created arithmetic operation: {left}")
            
            dbg_print(f"  Final Additive: {left}")
            return left
            
        return matcher(_match)

class Concat:
    @classmethod
    def match(cls):
        def _match(ts):
            dbg_print("\nConcat.match() - Starting")
            
            dbg_print("  Parsing left-hand side (Additive)")
            left = Additive.match()(ts)
            if left is NO_MATCH:
                dbg_print("  No left-hand side found")
                return NO_MATCH
            dbg_print(f"  Left-hand side: {left}")
            
            while True:
                op = ts.match(Amp)
                if op is NO_MATCH:
                    dbg_print("  No more '&' operators")
                    break
                    
                dbg_print("  Found '&' operator, parsing right-hand side")
                right = Additive.match()(ts)
                if right is NO_MATCH:
                    dbg_print("  No right-hand side found, breaking")
                    break
                    
                dbg_print(f"  Right-hand side: {right}")
                # Use literal '&' for the operator to avoid constructing an Amp token instance
                left = BinaryOperator('&', left, right)
                dbg_print(f"  Created concatenation: {left}")
            
            dbg_print(f"  Final Concat: {left}")
            return left
            
        return matcher(_match)

class Comparison:
    @classmethod
    def match(cls):
        def _match(ts):
            dbg_print("\nComparison.match() - Starting")
            
            dbg_print("  Parsing left-hand side (Concat)")
            left = Concat.match()(ts)
            if left is NO_MATCH:
                dbg_print("  No left-hand side found")
                return NO_MATCH
            dbg_print(f"  Left-hand side: {left}")
            
            # Check for comparison operators
            op_tokens = [
                (EqEq, '=='), (Neq, '!='), 
                (Lt, '<'), (Le, '<='), 
                (Gt, '>'), (Ge, '>=')
            ]
            
            dbg_print("  Checking for comparison operators")
            for op_token, op_str in op_tokens:
                dbg_print(f"    Trying {op_str} operator")
                if ts.match(op_token) is not NO_MATCH:
                    dbg_print(f"    Found {op_str} operator, parsing right-hand side")
                    right = Concat.match()(ts)
                    if right is NO_MATCH:
                        dbg_print("    No right-hand side found")
                        return NO_MATCH
                    dbg_print(f"    Right-hand side: {right}")
                    result = ComparisonOp(op_str, left, right)
                    dbg_print(f"    Created comparison: {result}")
                    return result
            
            dbg_print("  No comparison operator found, returning left-hand side")
            return left
            
        return matcher(_match)

class LogicalAnd:
    @classmethod
    def match(cls):
        def _match(ts):
            dbg_print("\nLogicalAnd.match() - Starting")
            
            dbg_print("  Parsing left-hand side (Comparison)")
            left = Comparison.match()(ts)
            if left is NO_MATCH:
                dbg_print("  No left-hand side found")
                return NO_MATCH
            dbg_print(f"  Left-hand side: {left}")
            
            while True:
                op = ts.match(And)
                if op is NO_MATCH:
                    dbg_print("  No more '&&' operators")
                    break
                    
                dbg_print("  Found '&&' operator, parsing right-hand side")
                right = Comparison.match()(ts)
                if right is NO_MATCH:
                    dbg_print("  No right-hand side found, rewinding")
                    ts.position -= 1  # Rewind the &&
                    break
                    
                dbg_print(f"  Right-hand side: {right}")
                left = LogicalOp('&&', left, right)
                dbg_print(f"  Created logical AND: {left}")
            
            dbg_print(f"  Final LogicalAnd: {left}")
            return left
            
        return matcher(_match)

class LogicalOr:
    @classmethod
    def match(cls):
        def _match(ts):
            dbg_print("\nLogicalOr.match() - Starting")
            
            dbg_print("  Parsing left-hand side (LogicalAnd)")
            left = LogicalAnd.match()(ts)
            if left is NO_MATCH:
                dbg_print("  No left-hand side found")
                return NO_MATCH
            dbg_print(f"  Left-hand side: {left}")
            
            while True:
                op = ts.match(Or)
                if op is NO_MATCH:
                    dbg_print("  No more '||' operators")
                    break
                    
                dbg_print("  Found '||' operator, parsing right-hand side")
                right = LogicalAnd.match()(ts)
                if right is NO_MATCH:
                    dbg_print("  No right-hand side found, rewinding")
                    ts.position -= 1  # Rewind the ||
                    break
                    
                dbg_print(f"  Right-hand side: {right}")
                left = LogicalOp('||', left, right)
                dbg_print(f"  Created logical OR: {left}")
            
            dbg_print(f"  Final LogicalOr: {left}")
            return left
            
        return matcher(_match)

class Expr:
    @classmethod
    def match(cls):
        return LogicalOr.match()

def parse_expression(text):
    dbg_print(f"\n=== parse_expression() ===")
    dbg_print(f"Input text: {text!r}")
    
    # Tokenize the input
    dbg_print("\n--- Tokenization ---")
    tokens = get_tokens(Token, text)
    dbg_print(f"Generated {len(tokens)} tokens:")
    for i, token in enumerate(tokens):
        dbg_print(f"  {i}: {token} (type: {type(token).__name__})")
    
    # Create token stream
    dbg_print("\n--- Parsing ---")
    ts = TokenStream(tokens)
    
    # Parse the expression
    result = Expr.match()(ts)
    
    dbg_print("\n--- Result ---")
    dbg_print(f"Parse result: {result}")
    if hasattr(result, 'to_tree'):
        dbg_print("AST tree:")
        dbg_print(result.to_tree())
    
    return result

