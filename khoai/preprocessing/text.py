

def create_ngrams(tokens, ngram=2):
    """Create n_grams"""
    new_tokens = []
    for i in range(len(tokens)-ngram):
        new_token = '_'.join(tokens[i:i+ngram])
        new_tokens.append(new_token)
    return new_tokens

