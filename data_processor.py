import time
import csv
import re
import numpy as np

from collections import Counter
from bs4 import BeautifulSoup
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class text_processor:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.ps = PorterStemmer()
        self.review_str = []
        self.review_ids = []
        self.labels_temp = []
        self.labels = []
        self.masks = []
        self.word_dict = {'UNK':0}
        self.label_dict = {'positive':1, 'negative':0}
        self.max_seq_len = 256

    def prepare_data(self, directory):
        self.read_csv(directory)
        self.build_vocab_dictionary()
        self.prepare_training_data()
        self.make_masks()
        self.pad_reviews()

    def read_csv(self, directory):
        with open(directory) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            list_iterator = iter(csv_reader)
            _ = next(list_iterator)

            for row in tqdm(list_iterator, total=50000):
            #for row, _ in zip(list_iterator, range(100)):
                self.review_str.append(self.clean_text(row[0]))
                self.labels_temp.append(self.label_dict[row[1]])

    def build_vocab_dictionary(self):
        flattened_review = [word for sentence in self.review_str for word in sentence]
        word_counter = Counter(flattened_review)
        word_counts = Counter({word:count for word, count in word_counter.items() if count>1}).most_common()

        for word, count in word_counts:
            self.word_dict[word] = len(self.word_dict)

    def prepare_training_data(self):
        for review, label in zip(self.review_str, self.labels_temp):
            if len(review) <= self.max_seq_len:
                self.review_ids.append([self.word_to_id(word) for word in review])
                self.labels.append(label)

    def make_masks(self):
        mask_indeces = [len(review) for review in self.review_ids]
        masks = self.fill_masks(mask_indeces)
        self.masks = masks

    def pad_reviews(self):
        self.review_ids = [review + [0]*(self.max_seq_len - len(review)) for review in self.review_ids]

    def fill_masks(self, seq_mask):
        number_line = np.arange(self.max_seq_len)
        mask_matrix = (number_line < np.expand_dims(np.array(seq_mask),1)).astype(float)
        mask_matrix[mask_matrix == 0] =  -1 * 10**10
        mask_matrix[mask_matrix == 1] = 0
        return mask_matrix

    def word_to_id(self, word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return self.word_dict['UNK']

    def clean_text(self, text):
        text = self.add_space_br(text)
        text = self.strip_html(text)
        text = self.remove_square_brackets(text)
        text = self.remove_special_characters(text)
        text = text.lower()
        word_list = self.tokenize_words(text)
        word_list = self.remove_stop_words(word_list)
        word_list = self.stem_words(word_list)
        return word_list

    def add_space_br(self, text):
        return re.sub('\>', '> ', text)

    def strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_square_brackets(self, text):
        return re.sub('\[.*\]', '', text)

    def remove_special_characters(self, text):
        pattern=r'[^a-zA-z0-9\s]'
        return re.sub(pattern,' ',text)

    def tokenize_words(self, text):
        return word_tokenize(text)

    def remove_stop_words(self, word_list):
        return [w for w in word_list if w not in self.stopwords]

    def stem_words(self, word_list):
        return [self.ps.stem(w) for w in word_list]
