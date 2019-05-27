def clean_entries(G_s, D_s):
    # defaulting to 1 if no value, strings for sym2transfer to handle
    if G_s[0] == "":
        G_s[0] = "1"
    if G_s[1] == "":
        G_s[1] = "1"
    if D_s[0] == "":
        D_s[0] = "1"
    if D_s[1] == "":
        D_s[1] = "1"

    # defining syntaxers and replacements
    syntaxers = {'^': '**', ')(': ')*(', 's(': 's*(', ')s': ')*s'}

    # making replacements for known syntaxes
    for key in syntaxers:
        G_s[0] = G_s[0].replace(key, syntaxers[key])
        G_s[1] = G_s[1].replace(key, syntaxers[key])
        D_s[0] = D_s[0].replace(key, syntaxers[key])
        D_s[1] = D_s[1].replace(key, syntaxers[key])

    # making replacements for "ns", value times s.
    for n in range(0, 9):
        # ns
        G_s[0] = G_s[0].replace(str(n) + "s", str(n) + "*s")
        G_s[1] = G_s[1].replace(str(n) + "s", str(n) + "*s")
        D_s[0] = D_s[0].replace(str(n) + "s", str(n) + "*s")
        D_s[1] = D_s[1].replace(str(n) + "s", str(n) + "*s")

        # sn
        G_s[0] = G_s[0].replace("s" + str(n), "s*" + str(n))
        G_s[1] = G_s[1].replace("s" + str(n), "s*" + str(n))
        D_s[0] = D_s[0].replace("s" + str(n), "s*" + str(n))
        D_s[1] = D_s[1].replace("s" + str(n), "s*" + str(n))

    return [G_s, D_s]

