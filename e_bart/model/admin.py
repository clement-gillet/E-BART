import collections

#just the g_encoder_attn in the decoder is randomly initialized.
def convert(state_dict):
    dict = state_dict

    pree = {k: dict[k] for k in list(dict)[0:1]} #shared / pas vrmt important
    pre_x = {k: dict[k] for k in list(dict)[1:3]} # preprocess pour x
    pre_gg = pre_x.copy() # preprocess pour g


    counter = 0

    while counter < len(pre_x.keys()):
        s1 = "encoder"
        s2 = "encoder_x"
        key = list(pre_x.keys())[0].replace(s1, s2)
        pre_x[key] = pre_x[list(pre_x.keys())[0]]
        del pre_x[list(pre_x.keys())[0]]
        counter += 1

    counter = 0

    while counter < len(pre_gg.keys()):
        s1 = "encoder"
        s2 = "encoder_g"
        key = list(pre_gg.keys())[0].replace(s1, s2)
        pre_gg[key] = pre_gg[list(pre_gg.keys())[0]]
        del pre_gg[list(pre_gg.keys())[0]]
        counter += 1

    counter = 0

    x = {k: dict[k] for k in list(dict)[3:179]}
    g = x.copy()

    x_head = {k: dict[k] for k in list(dict)[179:195]}
    g_head = x_head.copy()

    counter = 0

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
        s2 = "encoder_x"
        s3 = "11"
        s4 = "0"
        s5 = "layers"
        s6 = "head"
        key = list(x_head.keys())[0].replace(s1, s2)
        key = key.replace(s3, s4)
        key = key.replace(s5, s6)
        x_head[key] = x_head[list(x_head.keys())[0]]
        del x_head[list(x_head.keys())[0]]
        counter += 1

    counter = 0

    while counter < len(g_head.keys()):
        s1 = "encoder"
        s2 = "encoder_g"
        s3 = "11"
        s4 = "0"
        s5 = "layers"
        s6 = "head"
        key = list(g_head.keys())[0].replace(s1, s2)
        key = key.replace(s3, s4)
        key = key.replace(s5, s6)
        g_head[key] = g_head[list(g_head.keys())[0]]
        del g_head[list(g_head.keys())[0]]
        counter += 1

    post_encod_x = {k: dict[k] for k in list(dict)[195:197]}
    post_encod_gg = post_encod_x.copy()

    counter = 0

    while counter < len(post_encod_x.keys()):
        s1 = "encoder"
        s2 = "encoder_x"
        key = list(post_encod_x.keys())[0].replace(s1, s2)
        post_encod_x[key] = post_encod_x[list(post_encod_x.keys())[0]]
        del post_encod_x[list(post_encod_x.keys())[0]]
        counter += 1

    counter = 0

    while counter < len(post_encod_gg.keys()):
        s1 = "encoder"
        s2 = "encoder_g"
        key = list(post_encod_gg.keys())[0].replace(s1, s2)
        post_encod_gg[key] = post_encod_gg[list(post_encod_gg.keys())[0]]
        del post_encod_gg[list(post_encod_gg.keys())[0]]
        counter += 1

    decod = {k: dict[k] for k in list(dict)[197:]}

    counter = 0
    idx = 0
    while counter < len(decod.keys()) + 24:
        if ".encoder_attn." in list(decod.keys())[idx]:
            s1 = "encoder_attn"
            s2 = "x_encoder_attn"
            key = list(decod.keys())[idx].replace(s1, s2)
            decod[key] = decod[list(decod.keys())[idx]]
            del decod[list(decod.keys())[idx]]
        else:
            idx += 1
        counter += 1

    pretrained_ebart = {**pree, **pre_x, **pre_gg, **x, **g, **x_head, **g_head, **post_encod_x, **post_encod_gg, **decod}

    ebart = collections.OrderedDict(pretrained_ebart)

    # replace encoder_attn by x_encoder_attn
    # look like some are no initialized correctly while they are not new to BART... Why is that ? Verify !!
    # Now, we have all pieces so we should stitch it all together
    # verify that the weights are coorrect !!! I checked yesterday night and they were exact same !!!!!
    # one by one, inject in a new checkpoint file that we will save afterwards in a .pt file
    # And then that's it !!

    return {f'model.{k}': v for k, v in ebart.items()}
