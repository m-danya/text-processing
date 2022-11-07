import itertools
import re
from pathlib import Path


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


SENTENCE_DELIM = r'[.?!]'
SENTENCES_REGEXP = (
    r'(?P<sentence>(?<=' +
    SENTENCE_DELIM + r'\s).*?' +
    SENTENCE_DELIM +
    r'|' +
    r'^.*?' +
    SENTENCE_DELIM +
    r')'
)

# the script uses 40+ gb of RAM if not limited
# TL for N_LIMIT=1000
# TL for N_LIMIT=400
# N_LIMIT = 1000

# with open(Path(__file__).parent / 'russian_male_names.txt') as f:
#     russian_male_names = [name.strip() for name in f.readlines()]  # [:N_LIMIT]
# with open(Path(__file__).parent / 'russian_female_names.txt') as f:
#     russian_female_names = [name.strip() for name in f.readlines()]  # [:N_LIMIT]
# with open(Path(__file__).parent / 'russian_surnames.txt') as f:
#     russian_surnames = [name.strip() for name in f.readlines()]  # [:N_LIMIT]

# full_names = []

# useless if "Two Words" regex is used
# for surname in russian_surnames:
#     for name in itertools.chain(russian_male_names, russian_female_names):
#         full_names.append(r'\b' + name + r'\s' + surname + r'\b')

# full_names.extend(map(lambda s: r'\b' + s + r'\b', russian_male_names))
# full_names.extend(map(lambda s: r'\b' + s + r'\b', russian_female_names))
# full_names.extend(map(lambda s: r'\b' + s + r'\b', russian_surnames))

PERSONS_REGEXP = r'(?P<person>[А-Я][а-я]+\s[А-Я][а-я]+\s[А-Я][а-я]+|[А-Я][а-я]+\s[А-Я][а-я]+)'

# import re
# regexp = re.compile(PERSONS_REGEXP)
# for match in regexp.finditer('''
# Нургалиев уволил начальника УВД Томской области.
# Начальник УВД Томской области Виктор Гречман освобожден от занимаемой должности.
# Как сообщает "Интерфакс" со ссылкой на пресс-службу МВД, это решение принял глава
# ведомства Рашид Нургалиев по поручению президента РФ Дмитрия Медведева.
# '''):
#     print(f'type: {match.lastgroup}, span: {match.span()}')


def sg(s, name):
    # symbolic group
    return r'(?P<' + name + '>' + s + ')'

SERIES_NAME = '(?:(?:<td><h1 class="level2"><a(?: class="all"|) href="https:\/\/www\.kinopoisk\.ru\/series\/\d*\/\"(?: class="all"|)>)|' \
              '(?:<td><h1 class="level2"><a(?: class="all"|) href="\/series\/\d*\/\"(?: class="all"|)>))' +\
              sg('.*?', 'name') + r'(?=<\/a>)'
EPISODES = r'<td class=\"news\"><b>Эпизоды:<\/b><\/td>\s*<td>\s*<table(?: border="+0"+|)(?: cellspacing="+2"+|) cellpadding="+3"+(?: border="+0"+|)(?: cellspacing="+2"+|)>\s*(?:<tbody>|)\s*<tr>\s*<td class=\"news\">' + sg('.*?', 'episodes_count') + '(?=\<\/td\>)'

EPISODE_N = r'(?<=<span style=\"color:#777\">Эпизод )\d*'
EPISODE_NAME = r'(?<=style=\"font-size:16px;padding:0px;color:#444\"><b>).*?(?=<\/b>)'
EPISODE_OR_NAME = r'(?:class=[\'"]episodesOriginalName[\'"]>)' + sg('.*?', 'episode_original_name') + '<'
EPISODE_DATE1 = r'(?<=<td width=20% class="news" align=right valign=bottom style="border-bottom:1px dotted #ccc;padding:15px 0px;font-size:12px" align=left>).*?(?=</td>)'
EPISODE_DATE2 = r'(?<=<td class="news" style="border-bottom:1px dotted #ccc;padding:15px 0px;font-size:12px" width="20%" valign="bottom" align="right">).*?(?=</td>)'
EPISODE_DATE3 = r'(?<=align="left" class="news" style="border-bottom:1px dotted #ccc;padding:15px 0px;font-size:12px" valign="bottom" width="20%">).*?(?=</td>)'

SEASON_N = r'(?<=<h1 class="moviename-big" style="font-size:21px;padding:0px;margin:0px;color:#f60">Сезон )\d*(?=</h1>)'
SEASON_YEAR = r'\d\d\d\d(?=, эпизодов:)'
SEASON_EP = r'(?<=эпизодов: )\d*'
SERIES_REGEXP = r'|'.join(
    (
        SERIES_NAME,
        EPISODES,
        sg(EPISODE_N, 'episode_number'),
        sg(EPISODE_NAME, 'episode_name'),
        EPISODE_OR_NAME,
        sg(EPISODE_DATE1 + '|' + EPISODE_DATE2 + '|' + EPISODE_DATE3, 'episode_date'),
        sg(SEASON_N, 'season'),
        sg(SEASON_YEAR, 'season_year'),
        sg(SEASON_EP, 'season_episodes'),
    )
)
# print(SERIES_NAME)
# print(SERIES_REGEXP)
#
# regexp = re.compile(SERIES_REGEXP)
# entities = set()
# for match in regexp.finditer(html):
#     for key, value in match.groupdict().items():
#         if value is not None:
#             start, end = match.span(key)
#             entities.add((key, html[start:end]))
# for x in sorted(entities):
#     print(x)