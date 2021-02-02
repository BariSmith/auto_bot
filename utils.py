def filter_text(text):
    text = text.lower()
    text = list(filter(lambda x: x.isalpha() or x == '-', text))
    return ''.join(text)
