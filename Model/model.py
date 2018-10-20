import os
from os import walk
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f=[]
for (dirpath, dirnames, filenames) in walk(BASE_DIR + '/Dataset/Caracteres'):
    f.extend(filenames)
    break
f=[i[0] for i in f]
print(f)