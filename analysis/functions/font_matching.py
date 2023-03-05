cursive_dict = {
    'pml-rara': '$\\it{PML}$-$\\it{RARA}$',
    'npm1': '$\\it{NPM1}$',
    'cbfb-myh11': '$\\it{CBFB}$-$\\it{MYH11}$',
    'runx1-runx1t1': '$\\it{RUNX1}$-$\\it{RUNX1T1}$',
    'scd': 'Control',
    'stem cell donor': 'Control',
    'control': 'Control'
}


def edit(x, capitalize=True):
    x = x.lower()
    x = x.replace('aml-', '')
    for key, val in cursive_dict.items():
        x = x.replace(key, val)

    if(capitalize):
        x = x[:1].upper() + x[1:]
        x = x.replace('aml', 'AML')
    return x
