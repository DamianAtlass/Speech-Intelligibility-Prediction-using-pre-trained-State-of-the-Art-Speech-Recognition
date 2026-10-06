grid_kw_vocab = {
    "color": ['blue', 'green', 'red', 'white'], #4 items, index 1
    "letter": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z'], # 25 items, index 3
    "digit": ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero'] # 10 items, index 4
}

grid_vocab = {
    'command': {'bin', 'lay', 'place', 'set'}, #4
    'color': {'blue', 'green', 'red', 'white'}, #4
    'preposition': {'at', 'by', 'in', 'with'}, #4
    'letter': {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z'}, #25
    'digit': {'eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero'}, #10
    'adverb': {'again', 'now', 'please', 'soon'}, #4
}

grid_vocab_whisper_tokens = [2418, 3407, 586, 367, 277, 290, 1451, 538, 2182, 1081, 361, 283, 9505, 1732, 308, 257, 2360, 287, 412, 276, 262, 2309, 350, 732, 2031, 344, 741, 710, 4949, 2321, 5171, 288, 269, 3344, 256, 4018, 275, 365, 1045, 371, 992, 272, 1767, 297, 3092, 274, 797, 472, 280, 294, 3180]

grid_all_keywords = [x for v in grid_kw_vocab.values() for x in v]

grid_kw_indexes = [1, 3, 4]
grid_kw_labels = ["color", "letter", "digit"]
