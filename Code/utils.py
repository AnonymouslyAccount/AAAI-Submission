import os
import re
import time
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score, classification_report


def init_path(save_path, save_name):
    path_dict = {}

    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(save_path, save_name).mkdir(parents=True, exist_ok=True)

    path_dict['origin_output'] = os.path.join(save_path, save_name, 'output.txt')
    path_dict['processed_output'] = os.path.join(save_path, save_name, 'output.csv')
    path_dict['evaluate_output'] = os.path.join(save_path, save_name, 'metrics.txt')
    path_dict['config_output'] = os.path.join(save_path, save_name, 'config.json')
    path_dict['xlsx_output'] = os.path.join(save_path, save_name, 'metrics.xlsx')

    return path_dict


def dump_output(file, i, output, cost):
    file.write('==========\n')
    file.write(str(i + 1) + '\n')
    file.write(output + '\n')
    file.write(str(cost) + '\n')
    file.flush()


def dump_statistic(file, number, cost):
    file.write('======\n')
    file.write('Positive Number: %d\n' % number)
    file.write('Avg Time: %f\n' % cost)
    file.write('======\n')
    file.flush()


def check_output(output):
    label = 0
    output = output.split('Result')[-1]
    if output.find('include') != -1 and output.find('exclude') == -1:
        label = 1
    return label


def check_output_direct(output):
    if output.find('include') != -1:
        return 0
    return 1


def check_output_by_tag(output):
    result = re.findall('<result>(.*?)</result>', output.replace('\n', ''), re.IGNORECASE)
    if result:
        result = result[0].strip()
        if result.find('exclude') != -1:
            return 0
        return 1
    return 0


def intent_metric(x):
    cls_metric_dict = classification_report(x['gt'], x['predict'], output_dict=True)
    macro_metric, weighted_metric = cls_metric_dict['macro avg'], cls_metric_dict['weighted avg']
    metric_dict = {
        'acc': accuracy_score(x['gt'], x['predict']),
        'macro_p': macro_metric['precision'], 'macro_r': macro_metric['recall'], 'macro_f1': macro_metric['f1-score'],
        'weighted_p': weighted_metric['precision'], 'weighted_r': weighted_metric['recall'], 'weighted_f1': weighted_metric['f1-score'],
        'count': int(x.shape[0])
    }
    return pd.Series(metric_dict)


def get_intent_list(text, keywords, bipartite_graph):
    intent_list = set()

    keyword_list = get_keyword(text, keywords)
    keyword_list = keyword_list.split(',')

    for keyword in keyword_list:
        intent_list = intent_list.union(bipartite_graph[keyword])

    if "xxx" in intent_list:
        intent_list.remove("xxx")
    if len(intent_list):
        return list(intent_list)
    else:
        return ['general_prompt']


def get_keyword(text, keywords):
    keyword = []
    for key, value in keywords.items():
        for v in value:
            if text.find(v) != -1:
                keyword.append(key)
                break
    keyword = ','.join(keyword)
    return keyword


def get_exact_keyword(text, keywords):
    keyword = []
    for key, value in keywords.items():
        for v in value:
            if text.find(v) != -1:
                keyword.append(v)
                break
    keyword = list(set(keyword))
    keyword = ','.join(keyword)
    return keyword


def get_prior_label(intent, text, keywords):
    flag, flag_label = True, 0
    if (intent == 'xxx' and 'xx' in keywords) \
            or (intent == 'xxx' and 'xx' in keywords):
        flag, flag_label = False, 1
    return flag, flag_label



if __name__ == '__main__':
    pass
