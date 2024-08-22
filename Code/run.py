import argparse
from solver import Solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XiaoYi')

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--filter', action='store_true', default=False)
    parser.add_argument('--remove_lac', action='store_true', default=True)

    # Prompt
    parser.add_argument('--base_prompt_version', default='base_vx', type=str)
    parser.add_argument('--cluster_prompt_version', default='cluster_vx', type=str)
    parser.add_argument('--use_single_prompt', action='store_true', default=False)
    parser.add_argument('--use_direct_prompt', action='store_true', default=False)
    parser.add_argument('--use_specific_prompt', action='store_true', default=False)

    # Data
    parser.add_argument('--no_intent', action='store_true', default=False)
    # parser.add_argument('--data_path', default='balanced_valid_data_20240723_v3.3.csv', type=str)
    # parser.add_argument('--gt_path', default='balanced_valid_data_20240723_v3.3.csv', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--keyword_path', type=str)
    parser.add_argument('--keyword_to_intent_path', type=str)

    # LLM
    parser.add_argument('--model_name', default='Qwen2', type=str)
    parser.add_argument('--model_version', default='7B', type=str)

    # Evaluate
    parser.add_argument('--specific_intent', action='store_true', default=False)

    # Other
    parser.add_argument('--save_path', default='./IID', type=str)
    parser.add_argument('--save_name', default='', type=str)

    args = parser.parse_args()

    solver = Solver(args)
    if args.eval:
        solver.evaluate() if not args.no_intent else solver.evaluate_without_intent()
    else:
        solver.inference()
