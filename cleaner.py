def cleaner(dirt_line):

    #lower-case
    lowered = dirt_line.lower()

    #step1 remove all punctuation
    punct = ['\"', ',', '.', ':', ';', '?', '(', ')', '!']
    no_punct = []
    for ch in lowered:
        if ch not in punct:
            no_punct.append(ch)
    step1 = "".join(no_punct)

    clean = step1

    return clean
