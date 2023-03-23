import pathlib

from tqdm import tqdm

path_to_mistakes = pathlib.Path.cwd() / 'possible_tajik_misspellings.txt'
path_to_corpus = pathlib.Path.cwd() / 'preliminary_data' / 'final_filtered_newest.txt'
path_to_val_set = pathlib.Path.cwd() / 'final_data' / 'val_set.txt'

def main(path_to_mistakes=path_to_mistakes, path_to_corpus = path_to_corpus, path_to_val_set = path_to_val_set):

    with open(path_to_mistakes, encoding='utf-8') as f:
        mistakes = f.read().splitlines()
        mistakes = [elem.split('\t') for elem in mistakes]

    mistakes = {elem[0]: elem[1] for elem in mistakes}

    with open(path_to_corpus, encoding='utf-8') as f:
        sentences = f.read().splitlines()

    sentences_corrections = {}

    for sentence in tqdm(sentences):
        words = sentence.split(' ')
        for i in range(len(words)):
            if words[i] in mistakes:
                words[i] = mistakes[words[i]]
        corrected_sentence = ' '.join(words)
        if corrected_sentence != sentence:
            sentences_corrections[sentence] = corrected_sentence

    with open(path_to_val_set, 'w', encoding='utf-8') as f:
        f.write('mistaken\tcorrected\n')
        for sentence in sentences_corrections:
            f.write(sentence + '\t' + sentences_corrections[sentence] + '\n')

if __name__ == "__main__":
    main()
