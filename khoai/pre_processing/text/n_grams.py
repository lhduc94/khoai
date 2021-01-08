
def create_ngrams(tokens, ngram=2, determiner='_'):
    """Create n_grams.
    """
    new_tokens = []
    for i in range(len(tokens)-ngram+1):
        new_token = determiner.join(tokens[i:i+ngram])
        new_tokens.append(new_token)
    return new_tokens

