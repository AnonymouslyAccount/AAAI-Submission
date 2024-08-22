class ClusterPrompt:
    def __init__(self, base_version='base_v0529', cluster_version='cluster_v0529',
                 prompt_type='cluster'):
        self.clusters = {
            'cluster_0': [
                'xxx', 'xxx'
            ],
            # 'cluster_n': [
            #   'xxx', 'xxx', ...
            # ]
        }
        self.intent2cluster = {
            value: key for key, values in self.clusters.items() for value in values
        }

        self.direct = '''
Does the text contain personal information that needs recording, output "include" or "exclude"

Input: {text}
Output:
'''

        self.base_specific = '''
*** Your task is to determine whether the given input text contains the desired content based on several rules. If any one of the rules is satisfied, output "include"; otherwise, output "exclude."

You need to evaluate each rule and provide your reasoning and judgment result after evaluating all the rules. The reasoning should state which rule the given input satisfies, or if none are satisfied. The judgment result should state whether the desired content is included or not.

*** The rules are independent of each other. Determine which rule is satisfied, or if none are satisfied. 
{rule}

*** Output in the following format: 
Reasoning: Output "Satisfies rule X" or "None are satisfied" 
Judgment result: Output "include" or "exclude"

*** Note
The judgment result will be "include" if any rule in the list is satisfied.
Only if none of the rules are satisfied will the judgment result be "exclude."

*** Input: {text}

*** Output
'''

        self.base_vx = '''
### Task Description
Your task is to determine whether the input text contains the desired personal information based on the rules, and choose the judgment result from [include, exclude].

Below, several **positive rules** and **negative rules** are provided.
You need to evaluate each **positive rule** individually. If the text satisfies any one of them, the judgment result is 'include'; if none are satisfied, the result is 'exclude'.
You need to evaluate each **negative rule** individually. If the text satisfies any one of them, the judgment result is 'exclude'.

### Positive Rules
{rule}

### Negative Rules
{negative_rule}

### Output Format
Please strictly output in the following XML format without adding extra content:
<reason> Reasoning, satisfies rule X or none are satisfied, do not repeat the rule content </reason> 
<result>  Judgment result, include or exclude </result>

### Input
Input：{text}

### Output
'''

        general_rule = [
                    'xxx', 
                    'xxx',
                    # ...
                ]
        self.rule_specific_v0716 = {
            'general_prompt': {
                'rule': general_rule
            },
            '#intent#': {
                'rule': [
                    'xxx', 'xxx', '...'
                ],
                'negative_rule': [
                    'xxx', 'xxx', '...'
                ],
            },
            # ...
        }
        self.cluster_v0716 = {
            'general_prompt': {
                'rule': [
                    'xxx',
                    'xxx',
                    # '...'
                ]
            },
            'cluster_0': {
                'rule': [
                    'xxx', 'xxx', '...'
                ],
                'negative_rule': [
                    'xxx', 'xxx', '...'
                ],
                'example': [],
                'negative_example': []
            },
            'cluster_1': {
                'rule': [
                    'xxx', 'xxx', '...'
                ],
                'negative_rule': [
                    ''
                ],
                'example': [],
                'negative_example': []
            }
            # ...
        }
        
        self.base_prompt= self.get_base_prompt(base_version)
        self.cluster_prompt = self.get_cluster_prompt(cluster_version)

        self.prompt_type = prompt_type

    def get_base_prompt(self, version='base_vx'):
        return getattr(self, version)

    def get_cluster_prompt(self, version='cluster_vx'):
        return getattr(self, version)

    def get_full_prompt(self, text, intent):
        if self.prompt_type == 'direct':
            final_prompt = self.base_prompt.format(text=text)
        elif self.prompt_type == 'lac':
            final_prompt = self.base_prompt.format(text=text)
        else:
            intent = intent.lower().replace('_', ' ')

            try:
                if self.prompt_type == 'cluster':
                    cluster = self.intent2cluster[intent]
                elif self.prompt_type == 'single':
                    cluster = 'general_prompt'
                elif self.prompt_type == 'specific':
                    cluster = intent
                else:
                    cluster = None
            except KeyError:
                cluster = 'general_prompt'

            # print(cluster)
            cluster_info = self.cluster_prompt[cluster]
            rules = cluster_info['rule']
            rules = ['Rule %d - %s' % (i + 1, rule) for i, rule in enumerate(rules)]
            rules = '\n'.join(rules)

            negative_rules = cluster_info.get('negative_rule', 'None')
            if not len(negative_rules):
                negative_rules = 'None'
            if isinstance(negative_rules, list):
                negative_rules = ['Negative Rule %d - %s' % (i + 1, negative_rule) for i, negative_rule in enumerate(negative_rules)]
                negative_rules = '\n'.join(negative_rules)

            final_prompt = self.base_prompt.format(text=text, rule=rules,
                                                   negative_rule=negative_rules)

        return final_prompt


if __name__ == '__main__':
    prompts = ClusterPrompt()
    prompt = prompts.get_base_prompt('base_vx')
    print(prompts.intent2cluster)
