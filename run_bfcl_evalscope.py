import os
import re
import json
import argparse
import requests
from evalscope import TaskConfig, run_task


def sanitize_name(name: str) -> str:
    """
    把模型名转换成适合作为文件夹名的字符串。
    例如:
    Qwen/Qwen3.6-27B-GPTQ-Int4 -> Qwen_Qwen3.6-27B-GPTQ-Int4
    """
    name = name.strip()
    name = re.sub(r"[\\/:\*\?\"<>\|]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


def detect_model_name(api_url: str, api_key: str = "EMPTY") -> str:
    """
    从 OpenAI-compatible API 的 /v1/models 自动检测当前模型名称。
    """
    base = api_url.rstrip("/")
    models_url = base + "/models"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(models_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or not data["data"]:
            raise RuntimeError(f"/models 返回为空: {json.dumps(data, ensure_ascii=False)}")

        model_id = data["data"][0].get("id")
        if not model_id:
            raise RuntimeError(f"无法从 /models 返回中读取 id: {json.dumps(data, ensure_ascii=False)}")

        return model_id

    except Exception as e:
        raise RuntimeError(
            f"自动检测模型名称失败。请确认本地服务支持 {models_url}，"
            f"或者手动传入 --model-name。原始错误: {repr(e)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        default=None,
        help="可选。不填时自动从 /v1/models 检测当前模型名称。"
    )

    parser.add_argument(
        "--model-tag",
        default=None,
        help="可选。用于结果目录命名。不填时使用自动检测到的模型名。"
    )

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:1235/v1",
        help="OpenAI-compatible API 地址。"
    )

    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key。本地服务通常填 EMPTY 即可。"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="快速测试用 20/50；正式测试设为 0，表示不限制。"
    )

    parser.add_argument(
        "--output-dir",
        default="results_bfcl",
        help="结果输出目录。"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="评测温度。Agent / tool calling 对比建议设为 0。"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="最大输出 token 数。BFCL 多轮任务建议给大一些。"
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="你的 1235 端口一次只跑一个模型，本地评测建议 batch size 设为 1。"
    )

    parser.add_argument(
        "--is-fc-model",
        action="store_true",
        default=True,
        help="是否按原生 function calling 模型评测。默认 True。"
    )

    parser.add_argument(
        "--not-fc-model",
        action="store_true",
        help="如果你的本地 API 不支持 OpenAI tools/tool_calls，可以加这个参数。"
    )
    
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="可选。指定只跑哪些 BFCL-v3 subsets，例如 --subsets multi_turn_base multi_turn_miss_param"
    )


    args = parser.parse_args()

    if args.model_name:
        model_name = args.model_name
        print(f"使用手动指定模型名: {model_name}")
    else:
        model_name = detect_model_name(args.api_url, args.api_key)
        print(f"自动检测到当前模型名: {model_name}")

    model_tag = args.model_tag or model_name
    safe_model_tag = sanitize_name(model_tag)

    work_dir = os.path.join(args.output_dir, safe_model_tag)
    os.makedirs(work_dir, exist_ok=True)

    is_fc_model = args.is_fc_model
    if args.not_fc_model:
        is_fc_model = False

    dataset_args = {
        "bfcl_v3": {
            "subset_list": args.subsets if args.subsets else [
                "simple",
                "multiple",
                "parallel",
                "parallel_multiple",
                "java",
                "javascript",
                "live_simple",
                "live_multiple",
                "live_parallel",
                "live_parallel_multiple",
                "irrelevance",
                "live_relevance",
                "live_irrelevance",
                "multi_turn_base",
                "multi_turn_miss_func",
                "multi_turn_miss_param",
                "multi_turn_long_context",
            ],
            "extra_params": {
                "underscore_to_dot": True,
                "is_fc_model": is_fc_model,
            }
        }
    }


    generation_config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "parallel_tool_calls": True,
    }

    task_cfg = TaskConfig(
        model=model_name,
        api_url=args.api_url,
        api_key=args.api_key,
        eval_type="openai_api",
        datasets=["bfcl_v3"],
        eval_batch_size=args.eval_batch_size,
        dataset_args=dataset_args,
        generation_config=generation_config,
        limit=None if args.limit == 0 else args.limit,
        work_dir=work_dir,
    )

    print("BFCL-v3 评测配置:")
    print(json.dumps({
        "model": model_name,
        "model_tag": model_tag,
        "safe_model_tag": safe_model_tag,
        "api_url": args.api_url,
        "work_dir": work_dir,
        "limit": None if args.limit == 0 else args.limit,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "eval_batch_size": args.eval_batch_size,
        "is_fc_model": is_fc_model,
    }, ensure_ascii=False, indent=2))

    run_task(task_cfg)


if __name__ == "__main__":
    main()

