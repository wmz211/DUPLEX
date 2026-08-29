"""main.py —— EFI-Pilot CLI 入口。

模式：
  --mode hd   : 运行 HD 阶段（输入 JSON → 输出 HD Excel）
  --mode efi  : 运行 EFI 阶段（输入 HD Excel + JSON → 输出 EFI Excel）
  --mode eval : 运行评测（输入 HD Excel + EFI Excel + JSON → 输出三张评测表）

与原代码切换方法：
  原 HD:  python qwen_search.py
  新 HD:  python -m efi_pilot.main --mode hd [options]

  原 EFI: python predict_factuality.py [options]
  新 EFI: python -m efi_pilot.main --mode efi [options]
"""
import argparse
import os
from efi_pilot.utils.data_io import load_json
from efi_pilot.utils.environment import require_api_keys


def main():
    parser = argparse.ArgumentParser(description='EFI-Pilot 多智能体幻觉检测与事实性分析')
    parser.add_argument('--mode', choices=['hd', 'efi', 'eval'], required=True,
                        help='运行模式: hd / efi / eval')

    # 数据文件
    parser.add_argument('--json-cn',  default='hallu/extr-hallu-final.json',
                        help='中文 JSON（CD 文档）')
    parser.add_argument('--json-en',  default='hallu/en_experiment_documents2.json',
                        help='英文 JSON（ED 文档）')

    # HD 模式参数
    parser.add_argument('--hd-output', default='res/hd_results.xlsx',
                        help='HD 输出 Excel')
    parser.add_argument('--hd-log',    default='log/hd.txt')
    parser.add_argument('--hd-evidence', default=None,
                        help='HD intermediate evidence txt; default is <hd-output>_evidence.txt')
    parser.add_argument('--hd-calibration-records', default=None,
                        help='Structured Investigator CSV; default is <hd-output>_calibration.csv')
    parser.add_argument('--investigator-calibration', default=None,
                        help='JSON calibration file for Investigator alpha/threshold')
    parser.add_argument('--workers',   type=int, default=10,
                        help='HD 并行线程数（默认 10）')
    parser.add_argument('--log-mode',  choices=['r', 'o'], default='o',
                        help='日志模式: r=实时 / o=按序')

    # EFI 模式参数
    parser.add_argument('--hd-input',  default='res/hd_results.xlsx',
                        help='EFI 输入：HD 阶段的输出 Excel')
    parser.add_argument('--efi-output', default='res/efi_results.xlsx',
                        help='EFI 输出 Excel')
    parser.add_argument('--efi-log',   default='log/efi.txt')

    # 评测模式参数
    parser.add_argument('--eval-hd',  default='res/hd_results.xlsx')
    parser.add_argument('--eval-efi', default='res/efi_results.xlsx')

    # 公共参数
    parser.add_argument('--start-id', default=None)
    parser.add_argument('--end-id',   default=None)

    args = parser.parse_args()

    if args.mode == 'hd':
        _run_hd(args)
    elif args.mode == 'efi':
        _run_efi(args)
    elif args.mode == 'eval':
        _run_eval(args)


def _run_hd(args):
    from efi_pilot.orchestrator import run_hd
    api_keys = require_api_keys("deepseek", "bocha", "qwen")
    print(f"[HD 模式] 读取: {args.json_cn}")
    documents = load_json(args.json_cn)
    if os.path.exists(args.json_en):
        documents += load_json(args.json_en)
    print(f"共 {len(documents)} 个文档，使用 {args.workers} 个线程")

    config = {
        'deepseek_key':  api_keys["deepseek"],
        'bocha_key':     api_keys["bocha"],
        'qwen_key':      api_keys["qwen"],
        'max_workers':   args.workers,
        'realtime_mode': (args.log_mode != 'o'),
        'start_id':      args.start_id,
        'end_id':        args.end_id,
        'hd_evidence_file': args.hd_evidence,
        'hd_calibration_records': args.hd_calibration_records,
        'investigator_calibration': args.investigator_calibration,
    }
    os.makedirs(os.path.dirname(args.hd_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.hd_log), exist_ok=True)

    run_hd(documents, config, args.hd_output, args.hd_log)


def _run_efi(args):
    from efi_pilot.orchestrator import run_efi
    api_keys = require_api_keys("qwen")
    print(f"[EFI 模式] HD 输入: {args.hd_input}")
    docs_cn = load_json(args.json_cn)
    docs_en = load_json(args.json_en) if os.path.exists(args.json_en) else []

    config = {
        'qwen_key':     api_keys["qwen"],
        'start_id':     args.start_id,
        'end_id':       args.end_id,
    }
    os.makedirs(os.path.dirname(args.efi_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.efi_log), exist_ok=True)

    run_efi(args.hd_input, docs_cn, docs_en, config, args.efi_output, args.efi_log)


def _run_eval(args):
    from efi_pilot.evaluator import run_eval
    print(f"[Eval 模式] HD: {args.eval_hd} | EFI: {args.eval_efi}")
    run_eval(args.eval_hd, args.eval_efi, args.json_cn, args.json_en)


if __name__ == '__main__':
    main()
