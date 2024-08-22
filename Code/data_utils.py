import json
import pandas as pd


def read_input_data_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [json.loads(line) for line in lines]
    return lines


def read_input_data_from_csv(filename, specific=None, intent=None):
    file = pd.read_csv(filename, usecols=['intent', 'text'])
    if specific:
        file = file.loc[file['intent'].isin(intent), :]
        file = file.reset_index(drop=True)
    lines = [
        {'intent': file.iloc[i, 0], 'text': file.iloc[i, 1]} for i in file.index
    ]
    return lines


def read_input_data_from_xls(filename, specific=None, intent=None):
    file = pd.read_excel(filename)
    lines = [
        {'intent': None, 'text': file.iloc[i, 0]} for i in file.index
    ]
    return lines


def read_positive_data_from_csv(filename):
    file = pd.read_csv(filename, usecols=['intent', 'text', 'h_label'])
    lines = []
    for i in file.index:
        if file.iloc[i, 2] == 1:
            lines.append({'intent': file.iloc[i, 0], 'text': file.iloc[i, 1]})
    return lines


def read_keyword_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        keywords = {}
        for line in lines:
            if line.find(':') == -1:
                continue
            key, value = line.split(':')
            keywords[key] = [key] + [v.strip() for v in value.split(',')]
    return keywords


def read_bipartite_graph(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    keyword2intent = {}
    for key, values in data.items():
        for value in values:
            l = keyword2intent.get(value, [])
            l.append(key)
            keyword2intent[value] = l

    return keyword2intent


read_input_data = {
    '.txt': read_input_data_from_txt,
    '.csv': read_input_data_from_csv,
    '.xlsx': read_input_data_from_xls,
}
