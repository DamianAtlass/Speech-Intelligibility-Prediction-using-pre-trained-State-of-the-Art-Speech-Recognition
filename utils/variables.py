grid_kw_vocab = {
    "color": ['blue', 'green', 'red', 'white'], #4 items, index 1
    "letter": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z'], # 25 items, index 3
    "digit": ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero'] # 10 items, index 4
}

grid_all_keywords = [x for v in grid_kw_vocab.values() for x in v]

grid_kw_indexes = [1, 3, 4]
grid_kw_labels = ["color", "letter", "digit"]
