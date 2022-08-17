import argparse

opt = argparse.Namespace()
opt.ALPHABET = list('_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
opt.CHAR_LEN = len(opt.ALPHABET)
opt.ALPHABET_DICT = dict(zip(opt.ALPHABET, range(opt.CHAR_LEN)))
opt.MAX_CHAR_LEN = 4
