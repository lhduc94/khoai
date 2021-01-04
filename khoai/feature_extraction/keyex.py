class PosTagKeywordExtractor:
    """Keyword extraction."""

    def __init__(self, stopwords=None, min_len=1, max_len=10, determiner='__'):

        if stopwords is None:
            self.stopwords = []
        else:
            self.stopwords = stopwords
        self.min_len = min_len
        self.max_len = max_len
        self.determiner = determiner

    def combine_postagtype_(self, idx, tokens, postags, src_postags):
        t = [tokens[idx]]
        j = idx + 1
        while j < len(tokens) and postags[j] in src_postags:
            t.append(tokens[j])
            j += 1
        idx = j - 1
        if len(t) > 1:
            if t[-1] in self.stopwords:
                t = t[:-1]
        if len(t):
            new_token = self.determiner.join(t)
        else:
            new_token = None
        return idx, new_token

    def combine_postagtype(self, tokens, postags):
        new_tokens = list()
        new_postags = list()
        i = 0

        while i < len(tokens):
            if postags[i] in ['N', 'Np', 'Ny']:
                i, new_token = self.combine_postagtype_(i, tokens, postags, ['N', 'Np', 'Ny'])
                if new_token:
                    new_tokens.append(new_token)
                    new_postags.append('N')
            elif postags[i] in ['M']:
                i, new_token = self.combine_postagtype_(i, tokens, postags, ['M'])
                if new_token:
                    new_tokens.append(new_token)
                    new_postags.append('M')
            elif postags[i] in ['V']:
                i, new_token = self.combine_postagtype_(i, tokens, postags, ['V'])
                if new_token:
                    new_tokens.append(new_token)
                    new_postags.append('V')
            elif postags[i] in ['A']:
                i, new_token = self.combine_postagtype_(i, tokens, postags, ['A'])
                if new_token:
                    new_tokens.append(new_token)
                    new_postags.append('A')
            else:
                new_tokens.append(tokens[i])
                new_postags.append(postags[i])
            i += 1
        return new_tokens, new_postags

    def extract_phrase(self, tokens, postags, left_terms=None, right_terms=None):
        """Extract phrase."""
        if left_terms is None:
            left_terms = ['V', 'M']
        if right_terms is None:
            right_terms = ['A', 'V', 'M']
        s = []
        i = 0
        head = 0
        while i < len(tokens):
            if postags[i] == 'N' and postags[i] not in self.stopwords:
                t = [tokens[i]]
                start = i
                end = i
                if i == 1 and postags[0] in left_terms:
                    t = [tokens[0]] + t
                    start = 0
                elif i > head and postags[i - 1] in left_terms and tokens[i - 1] not in self.stopwords:
                    t = [tokens[i - 1]] + t
                    start = i - 1
                if i + 1 < len(tokens) and postags[i + 1] in right_terms and tokens[i + 1] not in self.stopwords:
                    t = t + [tokens[i + 1]]
                    end = i + 1
                    head = end
                    i = i + 1
                s.append((t, start, end))
            i = i + 1
        return s

    def len_phrase(self, phrase):
        phrase = phrase.replace(self.determiner, ' ').replace('_', ' ').split(' ')
        return len(phrase)

    def combine_phrases(self, phrases):
        """Combine Phrases."""
        phrase_2 = list()
        i = 0
        while i < len(phrases):
            t = phrases[i][0].copy()
            while i + 1 < len(phrases) and phrases[i][2] == phrases[i + 1][1]:
                i += 1
                t += phrases[i][0][1:]
            phrase_2.append(self.determiner.join(t))
            i += 1
        return [p for p in phrase_2 if self.min_len < self.len_phrase(p) < self.max_len]
