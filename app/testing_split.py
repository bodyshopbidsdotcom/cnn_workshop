from os import listdir
from os.path import join
from shutil import move
from random import shuffle

from constants import DATA, TESTING, IMAGE_FILE_SUFFIXES

from sklearn.model_selection import train_test_split


folder_a = DATA
folder_b = TESTING

files = listdir(DATA)
shuffle(files)
files = [file for file in files if file.lower().endswith(tuple(IMAGE_FILE_SUFFIXES))]

split = train_test_split(files, test_size=0.20)

for file in split[1]:
    move(join(folder_a, file), join(folder_b, file))
