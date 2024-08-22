import json
import os
import time
import pprint

import pandas as pd

import cluster
import data_utils
import model_utils
import utils
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm


class Solver:
    def __init__(self, args):
        self.args = args

        self.specific_intent = self.args.specific_intent

        print(f"Reading Data ......")
        self.input_data = \
            data_utils.read_input_data[os.path.splitext(args.data_path)[-1]](
                args.data_path, self.specific_intent, self.evaluate_intent)
        self.keyword = data_utils.read_keyword_from_txt(self.args.keyword_path)
        self.bipartite_graph = data_utils.read_bipartite_graph(self.args.keyword_to_intent_path)
        self.worthy_intent = None
        self.with_intent = not self.args.no_intent
        # self.gt_data = None

        print(f"Prepare Pipeline ......")
        check_output_dict = {
            'base_vx': utils.check_output_by_tag
        }
        self.need_lac = not self.args.remove_lac
        self.need_filter = self.args.filter
        if self.args.use_direct_prompt or self.args.use_lac_prompt:
            self.check_output = utils.check_output_direct
        else:
            self.check_output = check_output_dict.get(self.args.base_prompt_version, utils.check_output)

        print(f"Reading Prompt ......")
        self.prompt_type = 'cluster'
        if self.args.use_direct_prompt:
            self.prompt_type = 'direct'
            self.args.base_prompt_version = 'direct'
        if self.args.use_single_prompt:
            self.prompt_type = 'single'
        if self.args.use_specific_prompt:
            self.prompt_type = 'specific'
        if self.args.use_lac_prompt:
            self.prompt_type = 'lac'
            self.args.base_prompt_version = 'direct_lac'
        self.cluster_prompt = \
            cluster.ClusterPrompt(self.args.base_prompt_version, self.args.cluster_prompt_version,
                                  self.prompt_type)

        print(f"Checking Path ......")
        if not self.args.save_name:
            # self.args.save_name = time.strftime('%Y-%m-%d-%H-%M')
            self.args.save_name = '-'.join([
                self.args.model_name, self.args.model_version,
                self.args.base_prompt_version, self.args.cluster_prompt_version,
                self.prompt_type
            ])
        self.path_dict = utils.init_path(self.args.save_path, self.args.save_name)

        print(f"Creating Model ......")
        self.llm = getattr(model_utils, self.args.model_name)(version=self.args.model_version)
        # llm = Qwen2(version=7B)
        # cost, output = llm(prompt)

        self.__intent_graph_filter__()
        with open(self.path_dict['config_output'], 'a', encoding='utf-8') as file:
            json.dump(vars(self.args), file)

    def __pre_process__(self, intent, text):
        if self.need_lac and not utils.check_input_lac(text):
            return 0, False, 0
        else:
            return self.filter(intent, text) if self.need_filter else (0, True, 0)

    def __intent_graph_filter__(self):
        self.bipartite_graph_filtered = {}
        for key, values in self.bipartite_graph.items():
            value_graph = self.__intent_filter__(values)
            self.bipartite_graph_filtered[key] = value_graph

    def __intent_filter__(self, intent_list):
        clusters = set()
        intent_list_filtered = []
        for intent in intent_list:
            if intent == 'general_prompt':
                intent_list_filtered.append(intent)
                continue
            intent_cluster = self.cluster_prompt.intent2cluster[intent.lower().replace('_', ' ')]
            if intent_cluster not in clusters:
                clusters.add(intent_cluster)
                intent_list_filtered.append(intent)
        return intent_list_filtered

    def filter(self, intent, text):
        flag = False
        flag_label = 0

        start = time.time()

        keywords = utils.get_keyword(text, self.keyword)
        keywords = keywords.split(',')
        if '' not in keywords:
            # prior filter
            flag, flag_label = utils.get_prior_label(intent, text, keywords)

        end = time.time()
        cost = end - start

        return cost, flag, flag_label

    def inference(self):
        positive_data = 0
        total_time = 0
        total_data = {'intent': [], 'text': [], 'label': []}
        origin_file = open(self.path_dict['origin_output'], 'a', encoding='utf-8')

        for i, data in enumerate(tqdm(self.input_data)):
            intent, text = data['intent'], data['text']

            if intent:
                intent_list = [intent]
            else:
                intent_list = utils.get_intent_list(text, self.keyword, self.bipartite_graph_filtered)

            intent_list = self.__intent_filter__(intent_list)

            label = 0
            for intent in intent_list:

                cost, flag, flag_label = self.__pre_process__(intent, text)

                if flag:
                    prompt = self.cluster_prompt.get_full_prompt(text, intent)
                    llm_cost, output = self.llm(prompt)
                    cost += llm_cost
                    label = self.check_output(output)
                else:
                    output = 'Filtered'
                    label = flag_label

                total_time += cost
                utils.dump_output(origin_file, i, output, cost)

                if label:
                    break

            if label:
                positive_data += 1
            total_data['intent'].append(intent)
            total_data['text'].append(text)
            total_data['label'].append(label)

        origin_file.close()

        total_data = pd.DataFrame(total_data)
        total_data.to_csv(self.path_dict['processed_output'], encoding='utf-8-sig')
        with open(self.path_dict['evaluate_output'], 'a', encoding='utf-8') as file:
            avg_cost = total_time / len(self.input_data)
            utils.dump_statistic(file, positive_data, avg_cost)

        if self.with_intent:
            self.evaluate()
        else:
            self.evaluate_without_intent()

    def __evaluate_specific_intent__(self, data):
        data = data.loc[data['intent'].isin(self.evaluate_intent), :]
        return data

    def evaluate(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        file = open(self.path_dict['evaluate_output'], 'a', encoding='utf-8')
        file.write(f"==================== {self.args.save_name} =====================\n")

        predict = pd.read_csv(self.path_dict['processed_output'], usecols=['intent', 'text', 'label'])
        gt = pd.read_csv(self.args.gt_path, usecols=['intent', 'text', 'h_label'])

        if self.specific_intent:
            predict, gt = self.__evaluate_specific_intent__(predict), self.__evaluate_specific_intent__(gt)

        merged = pd.merge(gt, predict, on=['intent', 'text'], how='inner')
        merged = merged.rename(columns={'label': 'predict', 'h_label': 'gt'})

        file.write(f"Predict Number: {predict.shape[0]}\n")
        file.write(f"GT Number: {gt.shape[0]}\n")
        file.write(f"Merged Number: {merged.shape[0]}\n")

        total_acc = accuracy_score(merged['gt'], merged['predict'])
        file.write(f"Total Acc: {total_acc}\n")

        total_cls_report = classification_report(
            merged['gt'], merged['predict'], digits=4
        )
        file.write(f"Total Classification Report: \n")
        file.write(f"{total_cls_report}")

        intent_acc = merged.groupby('intent').apply(utils.intent_metric)
        file.write(f"Intent Metrics: \n")
        file.write(f"{intent_acc}")
        file.write(f"\n")

        file.close()

        different = merged.loc[merged['predict'] != merged['gt'], :]
        different.to_csv(self.path_dict['processed_output'].replace('output.csv', 'diff_output.csv'), encoding='utf-8-sig')

    def evaluate_without_intent(self):
        file = open(self.path_dict['evaluate_output'], 'a', encoding='utf-8')
        file.write(f"==================== {self.args.save_name} =====================\n")

        predict = pd.read_csv(self.path_dict['processed_output'], usecols=['text', 'label'])
        gt = pd.read_excel(self.args.gt_path)
        gt = gt.rename(columns={'query': 'text', 'is_store': 'gt'})
        gt.loc[gt['gt'] == 'Y', 'gt'] = 1
        gt.loc[gt['gt'] == 'N', 'gt'] = 0
        gt['gt'] = gt['gt'].astype(int)
        merged = pd.merge(gt, predict, on=['text'], how='inner')
        merged = merged.rename(columns={'label': 'predict'})
        is_store_accuracy = accuracy_score(merged['gt'], merged['predict'])
        is_store_recall = recall_score(merged['gt'], merged['predict'], pos_label=1)

        file.write(f'Accuracy: {is_store_accuracy}\n')
        file.write(f'Recall: {is_store_recall}\n')

        file.close()

        different = merged.loc[merged['predict'] != merged['gt'], :]
        different.to_csv(self.path_dict['processed_output'].replace('output.csv', 'diff_output.csv'),
                         encoding='utf-8-sig')


if __name__ == '__main__':
    pass
