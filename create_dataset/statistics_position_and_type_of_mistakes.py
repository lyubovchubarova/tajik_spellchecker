import pathlib
import csv
from dataclasses import dataclass, field
from typing import Dict
from collections import defaultdict

from Levenshtein import distance
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from qwerty_positions import qwerty_positions_russian
from utils import create_distances_dict

int_distances_russian = create_distances_dict(qwerty_positions=qwerty_positions_russian)

@dataclass
class ErrorType:
    relative_positions: list = field(default_factory=lambda: [])
    relative_frequencies: list = field(default_factory=lambda: [])
    keyboard_error_distances: list = field(default_factory=lambda: [])

    def count_relative_values(self, subject_of_count):

        relative_dict = defaultdict(float)

        if subject_of_count == 'positions':
            subject_of_count = self.relative_positions
        elif subject_of_count == 'keyboard_distances':
            subject_of_count = self.keyboard_error_distances

        for elem in subject_of_count:
            relative_dict[elem[0]] += elem[1]

        sum_values = sum(list(relative_dict.values()))

        relative_dict = {elem: relative_dict[elem]/sum_values for elem in relative_dict}

        return relative_dict


def define_error_type(standard: str, error: str) -> str:
    "Defines the type of typo: replace, insert, delete, swap"
    if len(error) < len(standard):
        return 'delete'
    if len(error) > len(standard):
        return 'insert'
    if len(error) == len(standard):
        if set(error) == set(standard):
            return 'swap'
        else:
            return 'replace'

def longest_common_prefix(str1: str, str2: str) -> int:
    "Defines the position of the end of the longest common prefix"
    result = ""

    len_str1 = len(str1)
    len_str2 = len(str2)

    # Compare str1 and str2
    i = 0
    j = 0

    while i <= len_str1 - 1 and j <= len_str2 - 1:
        if (str1[i] != str2[j]):
            break

        result += str1[i]
        i += 1
        j += 1

    return i

def process_word(standard: str, error: str, distances: Dict = int_distances_russian):

    error_type = define_error_type(standard, error)
    absolute_error_position = longest_common_prefix(standard, error)
    relative_error_position = min(absolute_error_position + 1, len(standard))/len(standard)
    keyboard_error_distance = -1

    if error_type == 'replace':
        keyboard_error_distance = distances[standard[absolute_error_position]][error[absolute_error_position]]
    elif error_type == 'insert':
        print(standard, error)
        print(standard[absolute_error_position])
        keyboard_error_distance = distances[standard[absolute_error_position]][error[absolute_error_position]]

    return error_type, relative_error_position, keyboard_error_distance

def get_statistics():

    # counters for each error_type
    replace_type = ErrorType()
    swap_type = ErrorType()
    delete_type = ErrorType()
    insert_type = ErrorType()

    # frequency_dict

    frequency_dict_filepath = pathlib.Path.cwd() / 'preliminary_data' / 'freqrnc2011.csv'
    frequency_dict = {}
    average_frequency = [0, 0]

    with open(frequency_dict_filepath, encoding='utf-8') as frequency_data:
        csv_reader = csv.reader(frequency_data, delimiter = '\t')
        next(csv_reader) # Skip header

        for row in csv_reader:
            if row[0] not in frequency_dict:
                frequency_dict[row[0]] = float(row[2])
            else:
                frequency_dict[row[0]] += float(row[2])

            average_frequency[0] += float(row[2])
            average_frequency[1] += 1

    average_frequency = average_frequency[0]/average_frequency[1]

    typos_filepath = pathlib.Path.cwd() / 'preliminary_data' / 'orfo_and_typos.L1_5.csv'

    with open(typos_filepath, encoding='utf-8') as typos_data:
        csv_reader = csv.reader(typos_data, delimiter = ';')
        next(csv_reader)  # Skip header

        cnt_1_error = 0
        cnt_2_error = 0
        cnt_3andmore_error = 0

        for row in tqdm(csv_reader):
            standard, error = row[0], row[1]
            if standard in frequency_dict:
                weight = float(row[2]) * frequency_dict[standard]
            else:
                weight = float(row[2]) * average_frequency
            levenstain_distance = distance(standard, error)

            if levenstain_distance >= 3: # 3 and more errors
                cnt_3andmore_error += 1

            elif levenstain_distance == 2: # 2 errors
                cnt_2_error += 1

            else: # 1 error
                cnt_1_error += 1
                error_type, relative_error_position, keyboard_error_distance = process_word(standard, error, int_distances_russian)

                if error_type == 'replace':
                    replace_type.relative_frequencies.append(weight)
                    replace_type.relative_positions.append((relative_error_position, weight))
                    replace_type.keyboard_error_distances.append((keyboard_error_distance, weight))
                elif error_type == 'swap':
                    swap_type.relative_frequencies.append(weight)
                    swap_type.relative_positions.append((relative_error_position, weight))
                elif error_type == 'insert':
                    insert_type.relative_frequencies.append(weight)
                    insert_type.relative_positions.append((relative_error_position, weight))
                    insert_type.keyboard_error_distances.append((keyboard_error_distance, weight))
                elif error_type == 'delete':
                    delete_type.relative_frequencies.append(weight)
                    delete_type.relative_positions.append((relative_error_position, weight))

    # Count statistics and visualizing graphs
    distribution_types_of_errors = {
            'replace': sum(replace_type.relative_frequencies),
            'insert': sum(insert_type.relative_frequencies),
            'delete': sum(delete_type.relative_frequencies),
            'swap': sum(swap_type.relative_frequencies)
        }

    sum_distrubution = sum(list(distribution_types_of_errors.values()))

    # how the types of errors distrubute
    distribution_types_of_errors = {type_of_error: distribution_types_of_errors[type_of_error]/sum_distrubution
                                    for type_of_error in distribution_types_of_errors}

    # how positions in each types of error distrubute
    replace_position = replace_type.count_relative_values('positions')
    insert_position = insert_type.count_relative_values('positions')
    delete_position = delete_type.count_relative_values('positions')
    swap_position = swap_type.count_relative_values('positions')

    # how distances on keyboard distributed
    replace_distances = replace_type.count_relative_values('keyboard_distances')
    insert_distances = insert_type.count_relative_values('keyboard_distances')

    return distribution_types_of_errors, \
           replace_position, \
           insert_position, \
           delete_position, \
           swap_position, \
           replace_distances, \
           insert_distances, \
           cnt_1_error, \
           cnt_2_error, \
           cnt_3andmore_error

def generate_plot(x, y1, name_x, name_y, name_plot):
    fig, ax = plt.subplots()
    ax.plot(x, y1, color='#8B008B', linewidth=2)
    plt.fill_between(x, y1, 0, facecolor='#6A5ACD', interpolate=True, alpha=0.7)

    plt.scatter(x[-1], y1[-1], color='#8718a3', s=30, marker='o')
    plt.scatter(x[0], y1[0], color='#8718a3', s=30, marker='o')

    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_title(name_plot)
    plt.xticks(rotation=60, horizontalalignment='center',)
    ax.legend()
    fig.tight_layout()
    ax.grid(alpha=0.5)
    plt.show()

if __name__ == "__main__":

    distribution_types_of_errors, \
    replace_position, \
    insert_position, \
    delete_position, \
    swap_position, \
    replace_distances, \
    insert_distances, \
    cnt_1_error, \
    cnt_2_error, \
    cnt_3andmore_error = get_statistics()

    # #plot_general_distrubution
    # data = list(distribution_types_of_errors.values())
    # labels = list(distribution_types_of_errors.keys())
    # colors = plt.get_cmap('Blues')(np.linspace(0.6, 0.2, len(data)))
    #
    # fig, ax = plt.subplots()
    # ax.pie(data,
    #        labels=labels,
    #        colors=colors,
    #        autopct='%1.1f%%',
    #        radius=4,
    #        center=(4, 4),
    #        wedgeprops={"linewidth": 1, "edgecolor": "white"})
    #
    # ax.set(xlim=(0, 8),
    #        ylim=(0, 8))
    #
    # plt.show()
    # plt.savefig('general_distribution.png')
    #
    # # plot mistakes position distrubition
    # plt.style.use('_mpl-gallery')
    #
    # # make data:
    # x = list(replace_position.keys())
    # y = list(replace_position.values())

    with open('statistics.py', 'w') as f:
        f.write('distribution_types_of_errors = ' + str(distribution_types_of_errors) + '\n')
        f.write('replace_position = ' + str(replace_position) + '\n')
        f.write('insert_position = ' + str(insert_position) + '\n')
        f.write('delete_position = ' + str(delete_position) + '\n')
        f.write('swap_position = ' + str(swap_position) + '\n')
        f.write('replace_distances = ' + str(replace_distances) + '\n')
        f.write('insert_distances = ' + str(insert_distances) + '\n')
        f.write('cnt_1_error = ' + str(cnt_1_error) + '\n')
        f.write('cnt_2_error = ' + str(cnt_2_error) + '\n')
        f.write('cnt_3andmore_error = ' + str(cnt_3andmore_error) + '\n')


