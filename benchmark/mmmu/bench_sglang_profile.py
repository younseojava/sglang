"""
    Bench the sglang-hosted vLM with benchmark MMMU

    Usage:
        python benchmark/mmmu/bench_sglang.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl

    The eval output will be logged
"""

import argparse
import base64
import dataclasses
import os
import random
import torch
from io import BytesIO

from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    parse_multi_choice_response,
    prepare_samples,
)
from tqdm import tqdm

from sglang import Engine
from sglang.srt.conversation import generate_chat_conv
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs


def run_generation(server_args, eval_args, samples):

    out_samples, answer_dict = dict(), dict()

    backend = Engine(**dataclasses.asdict(server_args))
    sampling_params = get_sampling_params(eval_args)

    for sample in tqdm(samples):
        prompt = sample["final_input_prompt"]
        image = sample["image"]
        buff = BytesIO()
        image.save(buff, format="PNG")
        base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        prefix = prompt.split("<")[0]
        suffix = prompt.split(">")[1]
        request_dict = {
            "model": "",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prefix,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_str}"
                            },
                        },
                        {
                            "type": "text",
                            "text": suffix,
                        },
                    ],
                }
            ],
        }

        conv = generate_chat_conv(
            ChatCompletionRequest(**request_dict),
            template_name=server_args.chat_template,
        )
        prompt = conv.get_prompt()
        if image is not None:
            gen_out = backend.generate(
                prompt=prompt,
                image_data=conv.image_data,
                sampling_params=sampling_params,
            )["text"]

            response = gen_out
        else:  # multiple images actually
            if sample["question_type"] == "multiple-choice":
                all_choices = sample["all_choices"]
                response = random.choice(all_choices)

            else:
                response = "INVALID GENERATION FOR MULTIPLE IMAGE INPUTS"

        if sample["question_type"] == "multiple-choice":
            pred_ans = parse_multi_choice_response(
                response, sample["all_choices"], sample["index2ans"]
            )
        else:  # open question
            pred_ans = response
        out_samples[sample["id"]] = pred_ans

        # set ground truth answer
        answer_dict[sample["id"]] = {
            "question_type": sample["question_type"],
            "ground_truth": sample["answer"],
        }

    backend.shutdown()

    return out_samples, answer_dict


def eval_mmmu(args):
    server_args = ServerArgs.from_cli_args(args)
    eval_args = EvalArgs.from_cli_args(args)

    if server_args.chat_template is None:
        raise ValueError("Chat template must be provided for this benchmark")

    samples = prepare_samples(eval_args)
    # use 2.5% of original samples
    n_samples = len(samples) // 40
    # samples = samples[:n_samples]
    samples = samples[0:2]


    profile_memory = False
    activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
    profiler = torch.profiler.profile(activities=activities, with_stack=False)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=profile_memory) as prof:
    #     with torch.profiler.record_function("generation"):
    #         out_samples, answer_dict = run_generation(server_args, eval_args, samples)
    # 
    # if profile_memory == False:
    #     print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    # else:
    #     print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # measurement block
    start_event.record()
    profiler.start()

    out_samples, answer_dict = run_generation(server_args, eval_args, samples)

    profiler.stop()
    end_event.record()
    torch.cuda.synchronize()

    prof_file = 'mmmu_profile.trace.json'
    parent_dir = os.path.dirname(os.path.abspath(prof_file))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(prof_file)
    etime = start_event.elapsed_time(end_event) / 1000.
    # args.output_path = f"{args.model_path}_val_sglang.json"
    # save_json(args.output_path, out_samples)
    # eval_result(output_path=args.output_path, answer_dict=answer_dict)

    print(f'Mmmu evaluation elapsed time = {etime:.4f} sec')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args

    eval_mmmu(args)
