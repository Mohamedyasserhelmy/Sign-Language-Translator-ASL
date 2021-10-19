from spellchecker import SpellChecker


def Auto_Correct(word):
    mySpellChecker = SpellChecker()
    return mySpellChecker.correction(word)
