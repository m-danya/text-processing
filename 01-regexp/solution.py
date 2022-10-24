PASSWORD_REGEXP = r'(?=.*[A-Z].*)(?=.*[a-z].*)(?=.*\d.*)(?![a-zA-Z\d]*([^$%@#&*!?])(?:[a-zA-Z\d]|\1)*$)(?!.*(.)\2.*$)(?:[a-zA-Z^$%@#&*!?\d]*){8,}'


def nc(s):
    # non-capturing group
    return r'(?:' + s + r')'


N_255 = r'0*\d|0*\d\d|0*1\d\d|0*2[0-4]\d|0*25[0-5]'
N_100_percents = r'0*\d%|0*\d\d%|100%'
RGB_VALUE = N_255 + '|' + N_100_percents

RGB = r'rgb\(' + (r'\s*' + nc(RGB_VALUE) + r'\s*' + ',') * 2 + r'\s*' + nc(RGB_VALUE) + r'\s*' + r'\)'

HEX_NUMBER = r'[\da-fA-F]'
HEX_NUMBER_FULL = HEX_NUMBER * 2

HEX = r'#' + HEX_NUMBER_FULL * 3 + r'|' + '#' + HEX_NUMBER * 3

HSL_360 = r'0*\d|0*\d\d|0*[1-2]\d\d|0*3[0-5]\d|0*360'

HSL = r'hsl\(' + r'\s*' + nc(HSL_360) + r'\s*' + ',' + r'\s*' + nc(N_100_percents) + r'\s*,' + r'\s*' + nc(N_100_percents) + r'\s*' + r'\)'

COLOR_REGEXP = nc(RGB) + '|' + nc(HEX) + '|' + nc(HSL)

VARIABLE = r'\b[a-zA-Z_][a-zA-Z_\d]*\b'
NUMBER = r'\b\d+(?:\.\d*)?|\.\d+\b'
CONSTANT = r'\b(?:pi|e|sqrt2|ln2|ln10)\b'
FUNCTION = r'\b(?:sin|cos|tg|ctg|tan|cot|sinh|cosh|th|cth|tanh|coth|ln|lg|log|exp|sqrt|cbrt|abs|sign)\b'
OPERATOR = r'[*^/\-+]'
LEFT_P = r'\('
RIGHT_P = r'\)'


def sg(s, name):
    # symbolic group
    return r'(?P<' + name + '>' + s + ')'


EXPRESSION_REGEXP = r'|'.join(
    (
        sg(NUMBER, 'number'),
        sg(CONSTANT, 'constant'),
        sg(FUNCTION, 'function'),
        sg(OPERATOR, 'operator'),
        sg(LEFT_P, 'left_parenthesis'),
        sg(RIGHT_P, 'right_parenthesis'),
        sg(VARIABLE, 'variable'),
    )
)


DATES_REGEXP = r'\d\d/\d\d/\d\d\d\d|\d\d\.\d\d\.\d\d\d\d'

# import re
# for match in re.finditer(EXPRESSION_REGEXP, 'sin(x) + cos(y) * 2.5'):
#     print(f'type: {match.lastgroup}, span: {match.span()}')
#
#
