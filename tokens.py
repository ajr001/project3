from matcher import NO_MATCH, EOF, matcher, Stream  # unified primitives


def dbg_print(*args):
    pass


class Token:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        cls_name = self.__class__.__name__
        if cls_name == '_T':
            # For special characters, show their value in quotes
            return f"{cls_name}('{self.value}')"
        else:
            # For other tokens, show the class name and value
            return f"{cls_name}({self.value!r})"

    @classmethod
    def match_zero_or_more(cls, matcher_fn):
        def _m(stream):
            ret = []
            while True:
                mark = stream.position
                result = matcher_fn(stream)
                if result is NO_MATCH:
                    break
                if mark == stream.position:
                    break
                ret.append(result)
            return ret
        return _m
    @classmethod
    def match_one_or_more(cls, matcher_fn):
        def _m(stream):
            first = matcher_fn(stream)
            ret = NO_MATCH
            if first is not NO_MATCH:
                rest = cls.match_zero_or_more(matcher_fn)(stream)
                ret = [first] + rest
            return ret
        return _m
    @classmethod
    def match_char(cls, ch):
        @matcher
        def _m(stream):
            ret = NO_MATCH
            c = stream.next()
            if c is not EOF and c == ch:
                ret = c
            return ret
        return _m

class Int(Token):
    @classmethod
    def match_digit(cls):
        @matcher
        def _m(stream):
            ret = NO_MATCH
            c = stream.next()
            if c is not EOF and c.isdigit():
                ret = c
            return ret
        return _m
    @classmethod
    def match(cls):
        digit = cls.match_digit()
        @matcher
        def _match(stream):
            ret = NO_MATCH
            digits = cls.match_one_or_more(digit)(stream)
            if digits is not NO_MATCH:
                ret = cls(''.join(digits))
            return ret
        return _match

class Bool(Token):
    def __str__(self):
        return 'true' if self.value else 'false'
    
    @classmethod
    def match(cls):
        @matcher
        def _match(stream):
            mark = stream.position
            true_str = 'true'
            false_str = 'false'
            
            # Try matching 'true'
            matched = True
            for c in true_str:
                if stream.next() != c:
                    matched = False
                    break
            
            if matched:
                return cls(True)
                
            # If 'true' didn't match, try 'false'
            stream.position = mark
            matched = True
            for c in false_str:
                if stream.next() != c:
                    matched = False
                    break
                    
            if matched:
                return cls(False)
                
            stream.position = mark
            return NO_MATCH
            
        return _match

class Null(Token):
    def __init__(self):
        super().__init__('null')
    
    @classmethod
    def match(cls):
        @matcher
        def _match(stream):
            mark = stream.position
            null_str = 'null'
            
            for c in null_str:
                if stream.next() != c:
                    stream.position = mark
                    return NO_MATCH
                    
            return cls()
            
        return _match

class String(Token):
    """String literal token that preserves the original quote character (single or double)."""
    def __init__(self, value, quote='"'):
        super().__init__(value)
        self.quote = quote  # Remember whether original used ' or "

    def __str__(self):
        escaped = self.value.replace(self.quote, f"\\{self.quote}")
        return f"{self.quote}{escaped}{self.quote}"
        
    @classmethod
    def match(cls):
        @matcher
        def _match(stream):
            mark = stream.position
            quote = stream.next()
            
            # Only match if it's a quote character
            if quote not in ('"', "'"):
                return NO_MATCH
                
            chars = []
            escaped = False
            
            while True:
                c = stream.next()
                
                if c is EOF:
                    stream.position = mark
                    return NO_MATCH
                    
                if escaped:
                    # Handle escape sequences
                    if c == 'n':
                        chars.append('\n')
                    elif c == 't':
                        chars.append('\t')
                    elif c in ('"', "'"):
                        chars.append(c)
                    else:
                        # For any other escaped character, just include it as-is
                        chars.append('\\' + c)
                    escaped = False
                elif c == '\\':
                    # Start of an escape sequence
                    escaped = True
                elif c == quote:
                    # End of string
                    return cls(''.join(chars), quote)
                else:
                    # Regular character
                    chars.append(c)
                    
        return _match

class Identifier(Token):
    """Token for identifiers (variable names, function names, etc.)"""
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return self.value
        
    @classmethod
    def match_char(cls):
        @matcher
        def _match(stream):
            c = stream.next()
            if c is EOF:
                return NO_MATCH
            if c.isalpha():
                return c
            return NO_MATCH
        return _match

    @classmethod
    def match(cls):
        @matcher
        def _match(stream):
            first = cls.match_char()(stream)
            if first is NO_MATCH:
                return NO_MATCH
                
            chars = [first]
            while True:
                c = stream.next()
                if c is EOF:
                    break
                if c.isalnum() or c == '_':
                    chars.append(c)
                else:
                    stream.position -= 1
                    break
                    
            return cls(''.join(chars))
            
        return _match

class SpecialChar(Token):
    CHAR = None
    @classmethod
    def make(cls, ch):
        # Create a proper class name based on the character
        class_name = {
            '!': 'Bang',
            '+': 'Plus',
            '-': 'Minus',
            '*': 'Star',
            '/': 'Slash',
            '=': 'Eq',
            '<': 'Lt',
            '>': 'Gt',
            '^': 'Caret',
            '%': 'Percent',
            '(': 'LParen',
            ')': 'RParen',
            '[': 'LBrack',
            ']': 'RBrack',
            ',': 'Comma',
            ':': 'Colon',
            '.': 'Dot',
            '&': 'Amp',
        }.get(ch, f'Char_{ord(ch)}')
        
        # Create the class with the proper name
        new_class = type(class_name, (cls,), {'CHAR': ch})
        return new_class
    @classmethod
    def match(cls):
        ch = cls.CHAR
        @matcher
        def _match(stream):
            ret = NO_MATCH
            c = stream.next()
            if c is not EOF and c == ch:
                ret = cls(c)
            return ret
        return _match

class MultiCharToken(Token):
    """Base class for multi-character tokens like !=, <=, >=, &&, ||"""
    def __init__(self, value):
        super().__init__(value)
    
    @classmethod
    def make(cls, value):
        def match(cls):
            @matcher
            def _match(stream):
                mark = stream.position
                for c in value:
                    if stream.next() != c:
                        stream.position = mark
                        return NO_MATCH
                return cls(value)
            return _match
            
        return type(value, (cls,), {
            "__str__": lambda self: self.value,
            "match": classmethod(match)
        })

# Single character tokens
LParen = SpecialChar.make('(')
RParen = SpecialChar.make(')')
LBrack = SpecialChar.make('[')
RBrack = SpecialChar.make(']')
Comma = SpecialChar.make(',')
Colon = SpecialChar.make(':')
Dot = SpecialChar.make('.')
Amp = SpecialChar.make('&')
Plus = SpecialChar.make('+')
Minus = SpecialChar.make('-')
Star = SpecialChar.make('*')
Slash = SpecialChar.make('/')
Eq = SpecialChar.make('=')
Lt = SpecialChar.make('<')
Gt = SpecialChar.make('>')
Bang = SpecialChar.make('!')
Caret = SpecialChar.make('^')
Percent = SpecialChar.make('%')

# Multi-character tokens
EqEq = MultiCharToken.make('==')
Neq = MultiCharToken.make('!=')
Le = MultiCharToken.make('<=')
Ge = MultiCharToken.make('>=')
And = MultiCharToken.make('&&')
Or = MultiCharToken.make('||')

def get_tokens(cls, text):
    stream = Stream(text)
    # Multi-char tokens must come before single-char tokens that are their prefixes
    token_types = [
        # Multi-char tokens first (order matters for correct matching)
        EqEq,  # == must come before =
        Neq,  # != must come before !
        Le,    # <= must come before <
        Ge,    # >= must come before >
        And,   # && must come before &
        Or,    # || must come before |
        # Single-char tokens
        Int, String, Bool, Null, Identifier,
        LParen, RParen, LBrack, RBrack, Comma, Colon, Dot, Amp,
        Plus, Minus, Star, Slash, Eq, Lt, Gt, Bang, Caret, Percent
    ]
    tokens = []
    while stream.position < len(stream.sequence):
        # Skip whitespace
        while stream.position < len(stream.sequence) and stream.sequence[stream.position].isspace():
            stream.position += 1
            
        if stream.position >= len(stream.sequence):
            break
            
        matched = False
        for token_type in token_types:
            mark = stream.position
            result = token_type.match()(stream)
            
            if result is not NO_MATCH:
                # Handle multi-char tokens
                if hasattr(token_type, 'value') and len(token_type.value) > 1:
                    # For multi-char tokens, we need to consume the correct number of characters
                    # The first character is already consumed by the matcher
                    for _ in range(len(token_type.value) - 1):
                        if stream.position < len(stream.sequence):
                            stream.position += 1
                
                # Handle special cases for keywords
                if isinstance(result, Identifier):
                    upper_v = result.value.upper()
                    if upper_v == 'TRUE':
                        result = Bool(True)
                    elif upper_v == 'FALSE':
                        result = Bool(False)
                    elif upper_v == 'NULL':
                        result = Null()
                
                dbg_print(f"  Created token: {result} (type: {type(result).__name__})")
                tokens.append(result)
                matched = True
                break
                
            # Reset position for next token type
            stream.position = mark
                
        if not matched:
            # If no token matched, raise an error
            current_char = stream.sequence[stream.position] if stream.position < len(stream.sequence) else 'EOF'
            raise ValueError(f"Unexpected character: '{current_char}' at position {stream.position}")
    
    return tokens
