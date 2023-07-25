
def convert(state_dict):
    dict = state_dict["model"]

    pre = {k: dict[k] for k in list(dict)[0:3]}
    x = {k: dict[k] for k in list(dict)[3:135]}
    g = x.copy()

    x_head = {k: dict[k] for k in list(dict)[135:147]}
    g_head = x_head.copy()

    counter=0

    while counter < len(x.keys()):
        s1 = "encoder"
        s2 = "encoder_x"
        key = list(x.keys())[0].replace(s1, s2)
        x[key] = x[list(x.keys())[0]]
        del x[list(x.keys())[0]]
        counter += 1

    counter = 0

    while counter < len(g.keys()):
        s1 = "encoder"
        s2 = "encoder_g"
        key = list(g.keys())[0].replace(s1, s2)
        g[key] = g[list(g.keys())[0]]
        del g[list(g.keys())[0]]
        counter += 1

    counter = 0

    while counter < len(x_head.keys()):
        s1 = "encoder"
        s2 = "document_head"
        s3 = "11"
        s4 = "0"
        key = list(x_head.keys())[0].replace(s1, s2)
        key = key.replace(s3, s4)
        x_head[key] = x_head[list(x_head.keys())[0]]
        del x_head[list(x_head.keys())[0]]
        counter += 1

    counter = 0

    while counter < len(g_head.keys()):
        s1 = "encoder"
        s2 = "guidance_head"
        s3 = "11"
        s4 = "0"
        key = list(g_head.keys())[0].replace(s1, s2)
        key = key.replace(s3, s4)
        g_head[key] = g_head[list(g_head.keys())[0]]
        del g_head[list(g_head.keys())[0]]
        counter += 1

    decod = {k: dict[k] for k in list(dict)[147:]}

    counter = 0
    idx = 0
    while counter < len(decod.keys())+10:
        if "encoder_attn" in list(decod.keys())[idx]:
            s1 = "encoder_attn."
            s2 = "x_encoder_attn."
            key = list(decod.keys())[idx].replace(s1, s2)
            decod[key] = decod[list(decod.keys())[idx]]
            del decod[list(decod.keys())[idx]]
        else:
            idx +=1
        counter += 1

    pretrained_ebart = {**pre, **x, **g, **x_head, **g_head, **decod}
    print(len(pre)) #
    print(len(x))
    print(len(g))
    print(len(x_head))
    print(len(g_head))
    print(len(decod))

    # replace encoder_attn by x_encoder_attn
    # look like some are no initialized correctly while they are not new to BART... Why is that ? Verify !!
    # Now, we have all pieces so we should stitch it all together
    # verify that the weights are coorrect !!! I checked yesterday night and they were exact same !!!!!
    # one by one, inject in a new checkpoint file that we will save afterwards in a .pt file
    # And then that's it !!

    '''
    print("-------------------------xxxxxxxxx--------------------")
    print(x.keys())
    print("-------------------------xxxxxxxxx--------------------")
    print("-------------------------gggggggggg--------------------")
    print(g.keys())
    print("-------------------------gggggggggg--------------------")
    print("-------------------------xheads--------------------")
    print(x_head.keys())
    print("-------------------------xheads--------------------")
    print("-------------------------gheads--------------------")
    print(g_head.keys())
    print("-------------------------gheads--------------------")
    print("-------------------------decod--------------------")
    print(decod)
    print("-------------------------decod--------------------")
    '''