#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: util.py
@time: 2019/3/9 15:43
"""
import json
import os
import spacy
import jieba


spacy_en = spacy.load('en')


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text.strip())]


def CN_tokenizer(text):
    return list(jieba.cut(text))


def dumpObj(obj, file):
    """
    dump object to file.
    :param obj:
    :param file:
    :return:
    """
    with open(file, 'w') as f:
        json.dump(obj, f)


def loadObj(file):
    """
    load object from file.
    :param file:
    :return:
    """
    with open(file, 'r') as f:
        obj = json.load(f)
    return obj


def ensure_dir(path):
    """
    ensure the dir exists.
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    text = "I like playing computer games."
    sent = "I want to watch tv in living room"
    print(tokenizer(text))
    print(tokenizer(sent))