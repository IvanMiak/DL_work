from os import listdir
from os.path import isfile, join
import ast
import astunparse
from sklearn.ensemble import RandomForestRegressor
import re
import string
import numpy as np
import pandas as pd
import _pickle as cPickle
import sys

def compare(inp, out, modelName):
    files1 = []
    files2 = []
    with open(inp, 'r') as f:
        filenames = f.readlines()
    for i in range(len(filenames)):
        with open(filenames[i].split()[0], 'r', encoding="utf8") as file:
            text = file.readlines()
            text = "".join(text)
            files1.append(astunparse.unparse(ast.parse(text)))
        with open(filenames[i].split()[1], 'r', encoding="utf8") as file:
            text = file.readlines()
            text = "".join(text)
            files2.append(astunparse.unparse(ast.parse(text)))
    columns = []
    l = len(files1)
    spec = ["for", "in", "if", "import", "return", "def", "self", "None", "print", "from", "with", "as", "True", "main",
            "init", "model"]
    punc = string.punctuation
    df0 = pd.DataFrame()
    df1 = pd.DataFrame()
    # частотность слов
    for w in spec:
        c = []
        for i in range(l):
            a = re.split(" ", ("".join([char for char in files[i] if char not in string.punctuation])).lower())
            c.append(a.count(w))
        c1 = np.array(c)
        df0["word " + w] = c1
        c = []
        for i in range(l):
            a = re.split(" ", ("".join([char for char in plagiat1[i] if char not in string.punctuation])).lower())
            c.append(a.count(w))
        c1 = np.array(c)
        df1["word " + w] = c1
        c = []
        columns.append("word " + w)
    # частотность символов
    for s in punc:
        c = []
        for i in range(l):
            c.append(files1[i].count(s))
        c1 = np.array(c)
        df0["symbol " + s] = c1
        c = []
        for i in range(l):
            c.append(files2[i].count(s))
        c1 = np.array(c)
        df1["symbol " + s] = c1
        columns.append("symbol " + s)
    # длина
    c = []
    for i in range(l):
        c.append(len(files1[i]))
    c1 = np.array(c)
    df0["len"] = c1
    c = []
    for i in range(l):
        c.append(len(files2[i]))
    c1 = np.array(c)
    df1["len"] = c1
    columns.append("len")
    # средняя длина слова
    c = []
    for i in range(l):
        c.append(len(files1[i]) / (len(files1[i].split()) + 0.001))
    c1 = np.array(c)
    df0["len word"] = c1
    c = []
    for i in range(l):
        c.append(len(files2[i]) / (len(files2[i].split()) + 0.001))
    c1 = np.array(c)
    df1["len word"] = c1
    columns.append("len word")
    # глубина конструкций
    c = []
    for i in range(l):
        max1 = 0
        ans = 0
        f = False
        for j in range(len(files1[i])):
            if files1[i][j] == '\n':
                f = True
            elif f and files1[i][j] == ' ':
                ans += 1
            else:
                f = False
                if ans > max1:
                    max1 = ans
                ans = 0
        c.append(max1)
    c1 = np.array(c)
    df0["dep"] = c1
    c = []
    for i in range(l):
        max1 = 0
        ans = 0
        f = False
        for j in range(len(files2[i])):
            if files2[i][j] == '\n':
                f = True
            elif f and files2[i][j] == ' ':
                ans += 1
            else:
                f = False
                if ans > max1:
                    max1 = ans
                ans = 0
        c.append(max1)
    c1 = np.array(c)
    df1["dep"] = c1
    columns.append("dep")
    X = pd.DataFrame()
    y = []

    def abs(x):
        if x < 0:
            x *= -1
        return x
    for i in range(l):
        x = pd.DataFrame()
        for column in columns:
            x[column] = np.array([abs(df0.loc[i, column] - df1.loc[j, column])])
        X = pd.concat([X, x])

    with open(modelName, 'rb') as f:
        forest = cPickle.load(f)
    y = forest.predict(X)
    with open(file2, 'w')as f:
        for i in range(len(y)):
            print(y[i], file=f)


if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    modelName = sys.argv[4]
    compare(file1, file2, modelName)
