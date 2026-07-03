import werpy
import num2words
import re
from typing import Union, List
import torch
# see https://pypi.org/project/werpy/

def calculate_wers_with_norm(
        reference: list[str],
        hypothesis: list[str],
) -> torch.Tensor:
    hypothesis = werpy.normalize(hypothesis)
    reference = werpy.normalize(reference)

    return torch.Tensor(werpy.wers(reference, hypothesis))

def normalize(strings: list[str],
              apply_separate_numbers_from_letter: bool = True,
              apply_numbers_to_words: bool = True,
              apply_werpy_normalize: bool = True
              ) -> list[str]:
    if apply_separate_numbers_from_letter:
        strings = [separate_numbers_from_letter(o) for o in strings]

    if apply_numbers_to_words:
        strings = [numbers_to_words(o) for o in strings]

    if apply_werpy_normalize:
        strings = [werpy.normalize(o) for o in strings]

    return strings


#print(werpy.wers("The cat is sleeping on the mat.", "The cat is playing on mat."))

def separate_numbers_from_letter(string: str) -> str:
    """
    Separates numbers from letters in a string by adding spaces.
    Example:
    "tree4wood10"         -> "tree 4 wood 10"
    "sdfa-.,`2**##123sdf" -> "sdfa-.,` 2 **## 123 sdf"

    Parameters
    ----------
    string: str

    Returns
    -------
    str

    """
    numbers_in_string = set(re.findall(r'\d+', string))

    # add space before number
    for n in numbers_in_string:
        insertions = 0
        occurrences_idx = [m.start() for m in re.finditer(n, string)]
        for o in occurrences_idx:
            o = o + insertions
            try:
                if o - 1 > 0:
                    character_before = string[o - 1]
                    if not character_before.isdigit() and not character_before == " ":
                        string = string[:o] + ' ' + string[o:]
                        insertions += 1
            except IndexError as e:
                pass

        insertions = 0
        occurrences_idx = [m.start() for m in re.finditer(n, string)]
        for o in occurrences_idx:
            o = o + insertions
            try:
                character_after = string[o + len(n)]
                if not character_after.isdigit() and not character_after == " ":
                    string = string[:o + len(n)] + ' ' + string[o + len(n):]
                    insertions += 1
            except IndexError as e:
                pass

    return string

def numbers_to_words(string : str) -> str:
    """
    Converts a string of numbers to words.
    Example:
    "2 Fast 2 Furious" -> 'two Fast two Furious'

    Parameters
    ---------

    string: str

    Returns
    ---------

    string: str
    """
    numbers_in_string = set(re.findall(r'\d+', string))
    for n in sorted(numbers_in_string, key= lambda x: len(x), reverse=True):
        string = string.replace(n, num2words.num2words(n))

    return string


if __name__ == '__main__':
    if True:
        ref = "set blue by h four again"
        hypo = " Set blue by H4 again."

        ref_norm = werpy.normalize(ref)
        hypo_norm = werpy.normalize(hypo)
        hypo_norm = numbers_to_words(separate_numbers_from_letter(hypo_norm))
        print(ref_norm)
        print(hypo_norm)

        print(werpy.wer(ref_norm, hypo_norm))




