
suffixes = set(open("suffixes.txt","r").read().split('\n'))
prefixes = set(open("prefixes.txt","r").read().split('\n'))

def stem(w):
    return remove_suffix(remove_prefix(w))

def remove_suffix(w):
    if w[-4:] in suffixes:
        return w[:-4]
    elif w[-3:] in suffixes:
        return w[:-3]
    elif w[-2:] in suffixes:
        return w[:-2]
    elif w[-1:] in suffixes:
        return w[:-1]
    else:
        return w

def remove_prefix(w):
    if w[:5] in prefixes:
        return w[5:]
    elif w[:4] in prefixes:
        return w[4:]
    elif w[:3] in prefixes:
        return w[3:]
    elif w[:2] in prefixes:
        return w[2:]
    elif w[:1] in prefixes:
        return w[1:]
    else:
        return w
