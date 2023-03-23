import csv
import pathlib
import re

from Levenshtein import distance
from collections import  defaultdict

from tqdm import tqdm


def collect_statistics():
    path_to_corpus = pathlib.Path.cwd() / 'preliminary_data' / 'final_filtered_newest.txt'

    len_corpus = 0
    words_occurences = defaultdict(int)

    with open(path_to_corpus, encoding='utf-8') as f:
        sentences = f.read().splitlines()

    for sentence in sentences:
        words = sentence.split()
        len_corpus += len(words)

        for word in words:
            if len(word) > 3:
                words_occurences[word] += 1

    words_occurences = {word: words_occurences[word]/len_corpus for word in words_occurences}

    return words_occurences

def find_mistakes(words_occurences):

    path_to_russian_mistakes = pathlib.Path.cwd() / 'preliminary_data' / 'orfo_and_typos.L1_5.csv'

    with open(path_to_russian_mistakes, encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter=';')

        next(csv_reader)

        len_mistakes = 0
        sum_mistakes = 0

        for row in csv_reader:
            len_mistakes += 1
            sum_mistakes += float(row[2])

    word_error_avgdiff = (sum_mistakes/len_mistakes)/2


    possible_misspellings = []

    keys_for_iteration = list(words_occurences.keys())

    for i in tqdm(range(len(keys_for_iteration) - 1)):
        for j in range(i+1, len(keys_for_iteration)):
            first_word = keys_for_iteration[i]
            second_word = keys_for_iteration[j]
            if first_word != second_word:
                if len(first_word)  > 5 and len(second_word) > 5 and\
                not(first_word in second_word) and\
                not(second_word in first_word) and\
                not (first_word[:-1] == second_word[:-1]):
                    if words_occurences[first_word]/words_occurences[second_word] <= word_error_avgdiff:
                        if distance(first_word, second_word) == 1:
                            possible_misspellings.append((first_word, second_word))


    return possible_misspellings

if __name__ == "__main__":

    tajik_dict_filepath = r'C:\Users\lyuba\PycharmProjects\generate_tadjik_misspelling\preliminary_data\lexemes.txt'
    pattern = re.compile(r'lex: (.+)\n')
    tajik_lexemes = set()
    with open(tajik_dict_filepath, encoding='utf-8') as f:
        tajik_dict_prelim = f.read().split('-')

        for elem in tajik_dict_prelim:
            lexeme = pattern.findall(elem)
            if lexeme:
                tajik_lexemes.add(lexeme[0])

    words_occurence = collect_statistics()
    possible_misspellings = find_mistakes(words_occurence)

    possible_misspellings = [(pair[0], pair[1])
                             for pair in possible_misspellings
                             if not (pair[0] in tajik_lexemes and pair[1] in tajik_lexemes)]

    print(len(possible_misspellings))

    with open('final_data/possible_tajik_misspellings.txt', 'w', encoding='utf-8') as f:
        f.write('MISSPELLED\tRIGHT\n')
        for pair in possible_misspellings:
            f.write(pair[0] + '\t' + pair[1] + '\n')



