"""Frozen Study 1 scoring functions, extracted verbatim from the study archive."""


import unicodedata


def normalize_anls(text):
    return " ".join(str(text).lower().split())


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def anls(prediction, answers):
    p = normalize_anls(prediction)
    if not p:
        return 0.0
    scores = []
    for answer in answers:
        a = normalize_anls(answer)
        if a:
            d = levenshtein(p, a) / max(len(p), len(a))
            scores.append(1 - d if d < 0.5 else 0.0)
    return max(scores, default=0.0)


def normalize_jp(value):
    value = unicodedata.normalize('NFKC', str(value)).lower()
    value = ' '.join(value.split()).rstrip('。').strip()
    if value.endswith('です'):
        value = value[:-2].rstrip('。').strip()
    return value


def score_value(corpus, prediction, status, answers, *, formatted):
    pred = str(prediction) if status in {'ok', 'truncated'} else ''
    if formatted and corpus == 'jdocqa':
        pred, answers = normalize_jp(pred), [normalize_jp(a) for a in answers]
    return anls(pred, answers)
