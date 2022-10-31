import itertools


def expand(s, root, n):
    if n == 0:
        return s.replace('X', '')
    return expand(s.replace('X', root), root, n - 1)


# X is to be replaced to ROOT_REGEXP
TYPE_1 = r'\(X\)'
TYPE_2 = r'\[X\]'
TYPE_3 = r'\{X\}'

ROOT = r'(?:' + r'|'.join((TYPE_1, TYPE_2, TYPE_3)) + r')*'

PARENTHESIS_REGEXP = expand(ROOT, ROOT, 9)


SENTENCE_DELIM = r'[.;:]'
SENTENCES_REGEXP = r'(?P<sentence>(?<=' + SENTENCE_DELIM + r')\s.*?' + SENTENCE_DELIM + r')'

from pathlib import Path

# the script uses 40+ gb of RAM if not limited
# TL for N_LIMIT=1000
N_LIMIT = 400

with open(Path(__file__).parent / 'russian_male_names.txt') as f:
    russian_male_names = [name.strip() for name in f.readlines()][:N_LIMIT]
with open(Path(__file__).parent / 'russian_female_names.txt') as f:
    russian_female_names = [name.strip() for name in f.readlines()][:N_LIMIT]
with open(Path(__file__).parent / 'russian_surnames.txt') as f:
    russian_surnames = [name.strip() for name in f.readlines()][:N_LIMIT]

full_names = []

for surname in russian_surnames:
    for name in itertools.chain(russian_male_names, russian_female_names):
        full_names.append(r'\b' + name + r'\s' + surname + r'\b')

full_names.extend(map(lambda s: r'\b' + s + r'\b', russian_male_names))
full_names.extend(map(lambda s: r'\b' + s + r'\b', russian_female_names))
full_names.extend(map(lambda s: r'\b' + s + r'\b', russian_surnames))
PERSONS_REGEXP = r'(?i)(?P<person>' + '|'.join(full_names) + ')'

# import re
# regexp = re.compile(PERSONS_REGEXP)
# for match in regexp.finditer('''
# Нургалиев уволил начальника УВД Томской области.
# Начальник УВД Томской области Виктор Гречман освобожден от занимаемой должности.
# Как сообщает "Интерфакс" со ссылкой на пресс-службу МВД, это решение принял глава
# ведомства Рашид Нургалиев по поручению президента РФ Дмитрия Медведева.
# '''):
#     print(f'type: {match.lastgroup}, span: {match.span()}')

SERIES_REGEXP = r''



