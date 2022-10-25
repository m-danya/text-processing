def nc(s):
    # non-capturing group
    return r'(?:' + s + r')'


def sg(s, name):
    # symbolic group
    return r'(?P<' + name + '>' + s + ')'


PASSWORD_REGEXP = r'(?=.*[A-Z].*)(?=.*[a-z].*)(?=.*\d.*)(?![a-zA-Z\d]*([^$%@#&*!?])(?:[a-zA-Z\d]|\1)*$)(?!.*(.)\2.*$)(?:[a-zA-Z^$%@#&*!?\d]*){8,}'

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

DAY = nc(r'0*[1-9]|0*[1-2]\d|0*3[0-1]')
MONTH = nc(r'0*[1-9]|1[0-2]')
YEAR = nc(r'\d+')

MONTH_RUS = nc(r'января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря')
MON_RUS = nc(r'янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек')
MONTH_ENG = nc(r'January|February|March|April|May|June|July|August|September|October|November|December')
MON_ENG = nc(r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec')

DATES_REGEXP = '|'.join(
    (
        nc(DAY + r'\.' + MONTH + r'\.' + YEAR),
        nc(DAY + r'/' + MONTH + r'/' + YEAR),
        nc(DAY + r'-' + MONTH + r'-' + YEAR),
        nc(YEAR + r'\.' + MONTH + r'\.' + DAY),
        nc(YEAR + r'/' + MONTH + r'/' + DAY),
        nc(YEAR + r'-' + MONTH + r'-' + DAY),
        nc(DAY + r'\s*' + MONTH_RUS + r'\s*' + YEAR),
        nc(MONTH_ENG + r'\s*' + DAY + r'\s*,\s*' + YEAR),
        nc(MON_ENG + r'\s*' + DAY + r'\s*,\s*' + YEAR),
        nc(YEAR + r'\s*,\s*' + MONTH_ENG + r'\s*' + DAY),
        nc(YEAR + r'\s*,\s*' + MON_ENG + r'\s*' + DAY),
    )
)
