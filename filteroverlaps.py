import csv
import collections
import lextrie

emolex = lextrie.LexTrie.from_plugin('emolex_en')
liwc = lextrie.LexTrie.from_plugin('liwc')
bing = lextrie.LexTrie.from_plugin('bing')

class StrictNgramDedupe(object):
    def __init__(self, data_path, ngram_size):
        self.ngram_size = ngram_size

        with open(data_path) as ip:
            rows = list(csv.DictReader(ip))
        self.data = rows
        self.work_matches = collections.defaultdict(list)

        for r in rows:
            self.work_matches[r['FAN_WORK_FILENAME']].append(r)

        # Use n-gram starting index as a unique identifier.
        self.starts_counter = collections.Counter(
            start
            for matches in self.work_matches.values()
            for start in self.to_ngram_starts(self.segment_full(matches))
        )

        filtered_matches = [self.top_ngram(span)
                            for matches in self.work_matches.values()
                            for span in self.segment_full(matches)]

        self.filtered_matches = [ng for ng in filtered_matches
                                 if self.no_better_match(ng)]

    def num_ngrams(self):
        return len(set(int(ng[0]['ORIGINAL_SCRIPT_WORD_INDEX'])
                       for ng in self.filtered_matches))

    def match_to_phrase(self, match):
        return ' '.join(m['ORIGINAL_SCRIPT_WORD'].lower() for m in match)

    def write_match_work_count_matrix(self, out_filename):
        ngrams = {}
        works = set()
        cells = collections.defaultdict(int)
        for m in self.filtered_matches:
            phrase = self.match_to_phrase(m)
            ix = int(m[0]['ORIGINAL_SCRIPT_WORD_INDEX'])
            filename = m[0]['FAN_WORK_FILENAME']

            ngrams[phrase] = ix
            works.add(filename)
            cells[(filename, phrase)] += 1

        ngrams = sorted(ngrams, key=ngrams.get)
        works = sorted(works)
        rows = [[cells[(fn, ng)] for ng in ngrams]
                for fn in works]
        totals = [sum(r[col] for r in rows) for col in range(len(rows[0]))]

        header = ['FILENAME'] + ngrams
        totals = ['(total)'] + totals
        rows = [[fn] + r for fn, r in zip(works, rows)]
        rows = [header, totals] + rows

        with open(out_filename, 'w', encoding='utf-8') as op:
            csv.writer(op).writerows(rows)

    def write_match_sentiment(self, out_filename):
        phrases = {}
        for m in self.filtered_matches:
            phrase = self.match_to_phrase(m)
            ix = int(m[0]['ORIGINAL_SCRIPT_WORD_INDEX'])
            phrases[phrase] = ix
        sorted_phrases = sorted(phrases, key=phrases.get)

        phrase_indices = [phrases[p] for p in sorted_phrases]
        phrases = sorted_phrases

        emo_count = [emolex.lex_count(p) for p in phrases]
        emo_sent_count = self.project_sentiment_keys(emo_count,
                                                     ['NEGATIVE', 'POSITIVE'])
        emo_emo_count = self.project_sentiment_keys(emo_count,
                                                    ['ANTICIPATION',
                                                     'ANGER',
                                                     'TRUST',
                                                     'SADNESS',
                                                     'DISGUST',
                                                     'SURPRISE',
                                                     'FEAR',
                                                     'JOY'])

        bing_count = [bing.lex_count(p) for p in phrases]
        bing_count = self.project_sentiment_keys(bing_count,
                                                 ['NEGATIVE', 'POSITIVE'])

        liwc_count = [liwc.lex_count(p) for p in phrases]
        liwc_sent_count = self.project_sentiment_keys(liwc_count,
                                                      ['POSEMO', 'NEGEMO'])
        liwc_other_keys = set(k for ct in liwc_count for k in ct.keys())
        liwc_other_keys -= set(['POSEMO', 'NEGEMO'])
        liwc_other_count = self.project_sentiment_keys(liwc_count,
                                                       liwc_other_keys)

        rows = self.compile_sentiment_groups(
            [emo_emo_count,
             emo_sent_count,
             bing_count,
             liwc_sent_count,
             liwc_other_count],
            ['NRC_EMOTION_',
             'NRC_SENTIMENT_',
             'BING_SENTIMENT_',
             'LIWC_SENTIMENT_',
             'LIWC_ALL_OTHER_']
        )

        for r, p, i in zip(rows, phrases, phrase_indices):
            r['{}-GRAM'.format(self.ngram_size)] = p
            r['{}-GRAM_START_INDEX'.format(self.ngram_size)] = i

        fieldnames = sorted(set(k for r in rows for k in r.keys()))
        totals = collections.defaultdict(int)
        skipkeys = ['{}-GRAM_START_INDEX'.format(self.ngram_size),
                    '{}-GRAM'.format(self.ngram_size)]
        totals[skipkeys[0]] = 0
        totals[skipkeys[1]] = '(total)'
        for r in rows:
            for k in r:
                if k not in skipkeys:
                    totals[k] += r[k]
        rows = [totals] + rows

        with open(out_filename, 'w', encoding='utf-8') as op:
            wr = csv.DictWriter(op, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(rows)

    def project_sentiment_keys(self, counts, keys):
        counts = [{k: ct.get(k, 0) for k in keys}
                  for ct in counts]
        for ct in counts:
            if sum(ct.values()) == 0:
                ct['UNDETERMINED'] = 1
            else:
                ct['UNDETERMINED'] = 0

        return counts

    def compile_sentiment_groups(self, groups, prefixes):
        new_rows = []
        for group_row in zip(*groups):
            new_row = {}
            for gr, pf in zip(group_row, prefixes):
                for k, v in gr.items():
                    new_row[pf + k] = v
            new_rows.append(new_row)
        return new_rows

    def get_spans(self, indices):
        starts = [0]
        ends = []
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                starts.append(i)
                ends.append(i)
        ends.append(len(indices))
        return list(zip(starts, ends))

    def segment_matches(self, matches, key):
        matches = sorted(matches, key=lambda m: int(m[key]))
        indices = [int(m[key]) for m in matches]
        return [[matches[i] for i in range(start, end)]
                for start, end in self.get_spans(indices)]

    def segment_fan_matches(self, matches):
        return self.segment_matches(matches, 'FAN_WORK_WORD_INDEX')

    def segment_orig_matches(self, matches):
        return self.segment_matches(matches, 'ORIGINAL_SCRIPT_WORD_INDEX')

    def segment_full(self, matches):
        return [orig_m
                for fan_m in self.segment_fan_matches(matches)
                for orig_m in self.segment_orig_matches(fan_m)
                if len(orig_m) >= self.ngram_size]

    def to_ngram_starts(self, match_spans):
        return [int(ms[i]['ORIGINAL_SCRIPT_WORD_INDEX'])
                for ms in match_spans
                for i in range(len(ms) - self.ngram_size + 1)]

    def start_count_key(self, span):
        def key(i):
            script_ix = int(span[i]['ORIGINAL_SCRIPT_WORD_INDEX'])
            return self.starts_counter.get(script_ix, 0)
        return key

    def no_better_match(self, ng):
        start = int(ng[0]['ORIGINAL_SCRIPT_WORD_INDEX'])
        best_start = max(range(start - self.ngram_size + 1,
                               start + self.ngram_size),
                         key=self.starts_counter.__getitem__)
        return start == best_start

    def top_ngram(self, span):
        start = max(
            range(len(span) - self.ngram_size + 1),
            key=self.start_count_key(span)
        )
        return span[start: start + self.ngram_size]


def process(ngram_size):
    filename = 'match-6gram-20170614-perfect.csv'
    out_prefix = 'force-awakens-most-common-perfect-matches-no-overlap-'
    matrix_out = '{}{}-gram-match-matrix.csv'.format(out_prefix, ngram_size)
    sentiment_out = '{}{}-gram-sentiment.csv'.format(out_prefix, ngram_size)

    dd = StrictNgramDedupe(filename, ngram_size=ngram_size)
    print(dd.num_ngrams())

    dd.write_match_work_count_matrix(matrix_out)
    dd.write_match_sentiment(sentiment_out)


if __name__ == '__main__':
    for ngram_size in [6, 8, 10]:
        process(ngram_size)
