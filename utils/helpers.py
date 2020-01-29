def get_number_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def printInfo(inputs):
    print('[Data Science Bowl 2018: Nuclei detection]: %s' % (inputs))

def presentParameters(args_dict):
    """
        Print the parameters setting line by line
        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    printInfo('=============== Parameters ===============')
    for key in sorted(args_dict.keys()):
        printInfo('{:>15} : {}'.format(key, args_dict[key]))
    printInfo('==========================================')
