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

def train(dir1, dir2, dir3, modelName):
    filenames = [f for f in listdir(dir1) if
                 isfile(join(dir1, f))]
    files = []
    plagiat1 = []
    plagiat2 = []
    for filename in filenames:
        with open(join(dir1, filename), 'r', encoding="utf8") as file:
            text = file.readlines()
            text = "".join(text)
            files.append(astunparse.unparse(ast.parse(text)))
        with open(join(dir2, filename), 'r', encoding="utf8") as file:
            text = "".join(text)
            plagiat1.append(astunparse.unparse(ast.parse(text)))
        with open(join(dir3, filename), 'r', encoding="utf8") as file:
            text = file.readlines()
            text = "".join(text)
            plagiat2.append(astunparse.unparse(ast.parse(text)))
    columns = []
    l = len(files)
    spec = ["for", "in", "if", "import", "return", "def", "self", "None", "print", "from", "with", "as", "True", "main",
            "init", "model"]
    punc = string.punctuation
    df0 = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
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
        for i in range(l):
            a = re.split(" ", ("".join([char for char in plagiat2[i] if char not in string.punctuation])).lower())
            c.append(a.count(w))
        c1 = np.array(c)
        df2["word " + w] = c1
        columns.append("word " + w)
    # частотность символов
    for s in punc:
        c = []
        for i in range(l):
            c.append(files[i].count(s))
        c1 = np.array(c)
        df0["symbol " + s] = c1
        c = []
        for i in range(l):
            c.append(plagiat1[i].count(s))
        c1 = np.array(c)
        df1["symbol " + s] = c1
        c = []
        for i in range(l):
            c.append(plagiat2[i].count(s))
        c1 = np.array(c)
        df2["symbol " + s] = c1
        columns.append("symbol " + s)
    # длина
    c = []
    for i in range(l):
        c.append(len(files[i]))
    c1 = np.array(c)
    df0["len"] = c1
    c = []
    for i in range(l):
        c.append(len(plagiat1[i]))
    c1 = np.array(c)
    df1["len"] = c1
    c = []
    for i in range(l):
        c.append(len(plagiat2[i]))
    c1 = np.array(c)
    df2["len"] = c1
    columns.append("len")
    # средняя длина слова
    c = []
    for i in range(l):
        c.append(len(files[i]) / (len(files[i].split()) + 0.001))
    c1 = np.array(c)
    df0["len word"] = c1
    c = []
    for i in range(l):
        c.append(len(plagiat1[i]) / (len(plagiat1[i].split()) + 0.001))
    c1 = np.array(c)
    df1["len word"] = c1
    c = []
    for i in range(l):
        c.append(len(plagiat2[i]) / (len(plagiat2[i].split()) + 0.001))
    c1 = np.array(c)
    df2["len word"] = c1
    columns.append("len word")
    # глубина конструкций
    c = []
    for i in range(l):
        max1 = 0
        ans = 0
        f = False
        for j in range(len(files[i])):
            if files[i][j] == '\n':
                f = True
            elif f and files[i][j] == ' ':
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
        for j in range(len(plagiat1[i])):
            if plagiat1[i][j] == '\n':
                f = True
            elif f and plagiat1[i][j] == ' ':
                ans += 1
            else:
                f = False
                if ans > max1:
                    max1 = ans
                ans = 0
        c.append(max1)
    c1 = np.array(c)
    df1["dep"] = c1
    c = []
    for i in range(l):
        max1 = 0
        ans = 0
        f = False
        for j in range(len(plagiat2[i])):
            if plagiat2[i][j] == '\n':
                f = True
            elif f and plagiat2[i][j] == ' ':
                ans += 1
            else:
                f = False
                if ans > max1:
                    max1 = ans
                ans = 0
        c.append(max1)
    c1 = np.array(c)
    df2["dep"] = c1
    columns.append("dep")
    X = pd.DataFrame()
    y = []

    def abs(x):
        if x < 0:
            x *= -1
        return x

    for i in range(l):
        for j in [i, (i + 1) % l, (i + 2) % l]:
            if i == j:
                y.append(0)
                y.append(0)
                y.append(0)
            else:
                y.append(1)
                y.append(1)
                y.append(1)
            x = pd.DataFrame()
            for column in columns:
                x[column] = np.array(
                    [abs(df0.loc[i, column] - df1.loc[j, column]), abs(df0.loc[i, column] - df2.loc[j, column]),
                     abs(df1.loc[i, column] - df2.loc[j, column])])
            X = pd.concat([X, x])
    y = np.array(y)

    forest = RandomForestRegressor(max_depth=20, n_estimators=10)
    forest.fit(X, y)
    with open(modelName, 'wb') as f:
        cPickle.dump(forest, f)


if __name__ == '__main__':
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    dir3 = sys.argv[3]
    modelName = sys.argv[5]
    train(dir1, dir2, dir3, modelName)


