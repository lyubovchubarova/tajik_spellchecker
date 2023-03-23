#import numpy as np
import pathlib
from collections import defaultdict
from typing import Dict, Tuple
from pprint import pprint
import random
from Levenshtein import distance
from tqdm import tqdm

from statistics import *

from utils import create_distances_dict
from qwerty_positions import qwerty_positions_tajik

tajik_distances = create_distances_dict(qwerty_positions_tajik)

### some human misspellings
tajik_distances['қ']['к'] = 1
tajik_distances['к']['қ'] = 1

tajik_distances['ҷ']['ч'] = 1
tajik_distances['ч']['ҷ'] = 1

tajik_distances['ӣ']['и'] = 1
tajik_distances['и']['ӣ'] = 1

tajik_distances['ҳ']['х'] = 1
tajik_distances['х']['ҳ'] = 1

tajik_distances['ғ']['г'] = 1
tajik_distances['г']['ғ'] = 1

tajik_distances['ӯ']['у'] = 1
tajik_distances['у']['ӯ'] = 1


replace_distances = {int(i): replace_distances[i] for i in replace_distances if int(i) <= 3}
insert_distances = {int(i): insert_distances[i] for i in insert_distances if int(i) <= 3}

def count_position(word, weight):
    return round(weight*(len(word) - 1))

def process_word_statistics(words, word_changed):

    type_of_the_change = random.choices(list(distribution_types_of_errors.keys()),
                                        weights=list(distribution_types_of_errors.values()),
                                        k=1)[0]

    if type_of_the_change == 'replace':
        position = count_position(words[word_changed],
                                  random.choices(list(replace_position.keys()),
                                                 weights=list(replace_position.values()),
                                                 k=1)[0]
                                  )
        symbol_distance = random.choices(list(replace_distances.keys()),
                                weights=list(replace_distances.values()),
                                k = 1)[0]

        cur_symbol = words[word_changed][position]
        symbol_for_replacement = random.choice([i for i in tajik_distances[cur_symbol] if
                                                tajik_distances[cur_symbol][i] == symbol_distance])

        words[word_changed] = words[word_changed][:position] + symbol_for_replacement + words[word_changed][position+1:]

    elif type_of_the_change == 'insert':
        position = count_position(words[word_changed],
                                  random.choices(list(insert_position.keys()),
                                                 weights=list(insert_position.values()),
                                                 k=1)[0]
                                  )
        symbol_distance = random.choices(list(insert_distances.keys()),
                                weights=list(insert_distances.values()),
                                k=1)[0]

        cur_symbol = words[word_changed][position]
        symbol_for_insertion = random.choice([i for i in tajik_distances[cur_symbol] if
                                                tajik_distances[cur_symbol][i] == symbol_distance])

        words[word_changed] = words[word_changed][:position] + symbol_for_insertion + words[word_changed][position:]

    elif type_of_the_change == 'swap':
        position = count_position(words[word_changed],
                                  random.choices(list(swap_position.keys()),
                                                 weights=list(swap_position.values()),
                                                 k=1)[0]
                                  )

        if position == len(words[word_changed]) - 1:
            position = position - 1


        words[word_changed] = words[word_changed][:position] +\
                              words[word_changed][position+1] +\
                              words[word_changed][position] +\
                              words[word_changed][position+2:]

    elif type_of_the_change == 'delete':
        position = count_position(words[word_changed],
                                  random.choices(list(delete_position.keys()),
                                                 weights=list(delete_position.values()),
                                                 k=1)[0]
                                  )

        words[word_changed] = words[word_changed][:position] + words[word_changed][position +1:]

def process_sentence_statistics(sentence):

    words = sentence.split(' ')

    changed_indexes = [i for i, word in enumerate(words) if len(word) > 2]

    if changed_indexes:

        word_changed = random.choice(changed_indexes)

        number_of_changes = random.choices([1, 2, 3], weights=[cnt_1_error, cnt_2_error, cnt_3andmore_error], k=1)[0]

        if number_of_changes == 1:
            process_word_statistics(words, word_changed)
        elif number_of_changes == 2:
            current_word = words[word_changed]
            while distance(current_word, words[word_changed]) < 2:
                process_word_statistics(words, word_changed)
                if len(words[word_changed]) <= 2:
                    break
        else:
            current_word = words[word_changed]
            while distance(current_word, words[word_changed]) < 3:
                process_word_statistics(words, word_changed)
                if len(words[word_changed]) <= 2:
                    break

    return ' '.join(words)

def process_word_random(words, word_changed):

    type_of_the_change = random.choice(list(distribution_types_of_errors.keys()))
    position = random.choice([i for i in range(len(words[word_changed]) - 1)])
    if type_of_the_change == 'replace':
        symbol_distance = random.choice(list(replace_distances.keys()))

        cur_symbol = words[word_changed][position]
        symbol_for_replacement = random.choice([i for i in tajik_distances[cur_symbol] if
                                                tajik_distances[cur_symbol][i] == symbol_distance])


        words[word_changed] = words[word_changed][:position] + symbol_for_replacement + words[word_changed][position+1:]

    elif type_of_the_change == 'insert':
        symbol_distance = random.choice(list(insert_distances.keys()))

        cur_symbol = words[word_changed][position]

        symbol_for_insertion = random.choice([i for i in tajik_distances[cur_symbol] if
                                                tajik_distances[cur_symbol][i] == symbol_distance])

        words[word_changed] = words[word_changed][:position] + symbol_for_insertion + words[word_changed][position:]

    elif type_of_the_change == 'swap':

        if position == len(words[word_changed]) - 1:
            position = position - 1

        words[word_changed] = words[word_changed][:position] +\
                              words[word_changed][position+1] +\
                              words[word_changed][position] +\
                              words[word_changed][position+2:]

    elif type_of_the_change == 'delete':
        words[word_changed] = words[word_changed][:position] + words[word_changed][position +1:]

def process_sentence_random(sentence):

    words = sentence.split(' ')

    changed_indexes = [i for i, word in enumerate(words) if len(word) > 2]

    if changed_indexes:
        word_changed = random.choice(changed_indexes)

        number_of_changes = random.choice([1, 2, 3])

        if number_of_changes == 1:
            process_word_random(words, word_changed)
        elif number_of_changes == 2:
            current_word = words[word_changed]
            while distance(current_word, words[word_changed]) < 2:
                process_word_random(words, word_changed)
                if len(words[word_changed]) <= 2:
                    break
        else:
            current_word = words[word_changed]
            while distance(current_word, words[word_changed]) < 3:
                process_word_random(words, word_changed)
                if len(words[word_changed]) <= 2:
                    break

    return ' '.join(words)


if __name__ == "__main__":

    stat_sentences = defaultdict(set)
    random_sentences = defaultdict(set)

    filename_dataset = pathlib.Path.cwd() / 'preliminary_data' / 'final_filtered_newest.txt'
    directory_to_datasets = pathlib.Path.cwd() / 'final_data'
    statistically_augmented_dataset_filename = directory_to_datasets / 'statistically_augmented_dataset.txt'
    randomly_augmented_dataset_filename = directory_to_datasets / 'randomly_augmented_dataset.txt'

    with open(statistically_augmented_dataset_filename, 'w', encoding='utf-8') as f:
        f.write('MISSPELLED\tCORRECTED\n')

    with open(randomly_augmented_dataset_filename, 'w', encoding='utf-8') as f:
        f.write('MISSPELLED\tCORRECTED\n')

    with open(filename_dataset, encoding='utf-8') as f:
        sentences = f.read().splitlines()

    const = 300000
    for i in range(8):
        for sentence in tqdm(sentences[i*const: min((i+1)*const, len(sentences))]):
            for j in range(2):
                worsened_statistics = process_sentence_statistics(sentence)
                if worsened_statistics != sentence:
                    stat_sentences[sentence].add(worsened_statistics)
                worsened_random = process_sentence_random(sentence)
                if worsened_random != sentence:
                    random_sentences[sentence].add(worsened_random)



        with open(statistically_augmented_dataset_filename, 'a', encoding='utf-8') as f:

            for sentence_good in stat_sentences:
                for sentence_worsened in stat_sentences[sentence_good]:
                    f.write(sentence_worsened + '\t' + sentence_good + '\n')

        with open(randomly_augmented_dataset_filename, 'a', encoding='utf-8') as f:

            for sentence_good in random_sentences:
                for sentence_worsened in random_sentences[sentence_good]:
                    f.write(sentence_worsened + '\t' + sentence_good + '\n')
