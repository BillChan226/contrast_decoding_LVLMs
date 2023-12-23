# the script generates captions for the images in the test set and save the captions
import os
import torch
import argparse
import numpy as np
import random
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from tqdm import tqdm
import json
from collections import defaultdict

# define pre-trained model download path by setting the environment variable
os.environ["TRANSFORMERS_CACHE"] = "./model_checkpoints/"
import transformers


def initialize_mini_gpt_4(parser, hyper_params):
    from transformers import StoppingCriteriaList
    from minigpt4.conversation.conversation import (
        Chat,
        CONV_VISION_Vicuna0,
        CONV_VISION_LLama2,
        StoppingCriteriaSub,
    )
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry



    args = parser.parse_args()

    decoding_strategy = args.decoder

    # load config
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id  # 0
    model_cls = registry.get_model_class(model_config.arch)  # minigpt4
    model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

    # available models
    conv_dict = {
        "pretrain_vicuna0": CONV_VISION_Vicuna0,
        "pretrain_llama2": CONV_VISION_LLama2,
    }
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [
        torch.tensor(ids).to(device="cuda:{}".format(args.gpu_id))
        for ids in stop_words_ids
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )

    chat = Chat(
        model,
        vis_processor,
        device="cuda:{}".format(args.gpu_id),
        stopping_criteria=stopping_criteria,
        decoding_strategy=decoding_strategy,
        hyper_params=hyper_params,
    )

    return chat, CONV_VISION, cfg


# main function
def generate_chair_input(hyper_params, args):

    decoding_strategy = args.decoder
    model_name = args.model_name
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    output_dir = args.output_dir
    num_samples = args.num_samples
    seed = args.seed
    verbosity = args.verbosity

    # print program level arguments
    if verbosity:
        print("\nDecoding strategy: ", decoding_strategy)
        print("backbone model_name: ", model_name)
        print("dataset_name: ", dataset_name)
        print("data_dir: ", data_dir)
        print("output_dir: ", output_dir)
        print("num_samples: ", num_samples)
        print("seed: ", seed)


    halc_params = hyper_params["halc_params"]
    context_domain = halc_params["context_domain"]
    contrast_weight = halc_params["contrast_weight"]
    context_window = halc_params["context_window"]
    expand_ratio = halc_params["expand_ratio"]

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load model
    if model_name == "minigpt4":
        model, CONV_VISION, cfg = initialize_mini_gpt_4(parser, hyper_params)

    if verbosity:
        print(f"\n{model_name} model initialized successfully.")

    # set output dir
    model_type = cfg.model_cfg.model_type.replace("_", "-")
    output_dir = os.path.join(
        output_dir, f"{model_name}_{model_type}", decoding_strategy, dataset_name
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # generated caption file path
    generated_captions_path = os.path.join(
        output_dir,
        f"{model_name}_{model_type}_{decoding_strategy}_{dataset_name}_{context_domain}_{contrast_weight}_{context_window}_{expand_ratio}_{num_samples}_generated_captions.json",
    )

    # chair input varies by dataset
    if dataset_name == "coco":
        annotation_file_path = os.path.join(
            data_dir,
            "annotations/captions_val2014.json",
        )
        # with the coco api
        coco = COCO(annotation_file_path)


        # if generated captions already exist
        if os.path.exists(generated_captions_path):
            # load the generated captions
            with open(generated_captions_path, "r") as f:
                all_generated_captions = json.load(f)
            if verbosity:
                print(f"\nLoaded generated captions from {generated_captions_path}.")
        else:
            # prepare data
            # all the image ids
            img_ids = coco.getImgIds()
            # sample image ids
            sampled_img_ids = random.sample(img_ids, num_samples)

            # generate captions
            all_generated_captions = []
            for i, cur_img_id in enumerate(
                tqdm(sampled_img_ids, desc="Generating Captions")
            ):
                cur_generated_captions_path = os.path.join(
                    output_dir,
                    f"{model_name}_{model_type}_{decoding_strategy}_{dataset_name}_{context_domain}_{contrast_weight}_{context_window}_{expand_ratio}_{i+1}_generated_captions.json",
                )

                # current image
                cur_img = coco.loadImgs(cur_img_id)[0]
                # current image path in the data dir
                cur_img_path = os.path.join(data_dir, cur_img["file_name"])
                # construct the conversation
                img_list = []
                model.upload_img(cur_img_path, CONV_VISION, img_list)
                
                model.encode_img(img_list, 38)  # -1 means the last layer
                # question taken from https://arxiv.org/pdf/2305.10355.pdf
                model.ask("Generate a one sentence caption of the image.", CONV_VISION)
                # model.ask("Generate a caption of the image with rich details.", CONV_VISION)
                output_text, _, _ = model.answer(
                    CONV_VISION,
                    img_list,
                    # dola_decoding=cfg.model_cfg.dola_decoding,
                )
                print("cur_img_path: ", cur_img_path)
                print("output_text: ", output_text)
                
                # append the generated caption to the list
                # format follows https://github.com/tylin/coco-caption/tree/master
                all_generated_captions.append(
                    {
                        "image_id": cur_img_id,
                        "caption": output_text,
                    }
                )

                # clear the chat
                CONV_VISION.messages = []

                with open(
                    cur_generated_captions_path,
                    "w",
                ) as f:
                    json.dump(all_generated_captions, f)

                if verbosity:
                    print(f"\nGenerated captions saved to {output_dir}.")

                # remove the previous file
                if i > 0:
                    prev_generated_captions_path = os.path.join(
                        output_dir,
                        # f"{model_name}_{model_type}_{decoding_strategy}_{dataset_name}_{i}_generated_captions.json",
                        f"{model_name}_{model_type}_{decoding_strategy}_{dataset_name}_{context_domain}_{contrast_weight}_{context_window}_{expand_ratio}_{i}_generated_captions.json",
                    )
                    os.remove(prev_generated_captions_path)

        # evaluate all the generated captions using coco-caption
        if verbosity:
            print("\nEvaluating generated captions...")

        # check the length of the generated captions
        loaded_json = json.load(open(generated_captions_path))
        # construct output file as input to CHAIR evaluation
        # output format follows https://github.com/ruotianluo/self-critical.pytorch
        formulated_output_dict = {}
        # overall result
        all_overall_scores = defaultdict(list)
        # imgToEval per image result
        img_to_eval_dict = {}
        # to save memory, load 100 captions at a time
        for start_idx in tqdm(
            range(0, len(loaded_json), 100), desc="Generating CHAIR Input"
        ):
            # define the current iteration end index
            end_idx = min(start_idx + 100, len(loaded_json))
            coco_res = coco.loadRes(
                loaded_json[start_idx:end_idx],
            )
            coco_eval = COCOEvalCap(coco, coco_res)
            coco_eval.params["image_id"] = coco_res.getImgIds()
            coco_eval.evaluate()

            # keep track of the overall scores
            for metric, score in coco_eval.eval.items():
                all_overall_scores[metric].append(score)

            # imgToEval per image result
            for i, cur_img_id in enumerate(coco_res.getImgIds()):
                cur_eval_dict = coco_eval.evalImgs[i]
                # add caption to the eval dict
                cur_eval_dict["caption"] = coco_res.imgToAnns[cur_img_id][0]["caption"]
                img_to_eval_dict[cur_img_id] = cur_eval_dict

        # overall result
        overall_dict = {}
        for metric, score in all_overall_scores.items():
            overall_dict[metric] = np.mean(score)
        formulated_output_dict["overall"] = overall_dict
        formulated_output_dict["imgToEval"] = img_to_eval_dict

        # sanity check the results
        if len(img_to_eval_dict) != num_samples:
            raise Exception(
                f"Resulting output_dict has number of images {len(img_to_eval_dict)} different from num_samples {num_samples}"
            )

        if verbosity:
            print(
                f"\nGenerated {len(img_to_eval_dict)} samples results in CHAIR format."
            )

        # save the formulated output dict
        formulated_output_path = os.path.join(
            output_dir,
            f"{model_name}_{model_type}_{decoding_strategy}_{dataset_name}_{context_domain}_{contrast_weight}_{context_window}_{expand_ratio}_{num_samples}_chair.json",

            # f"{model_name}_{model_type}_{decoding_strategy}_{dataset_name}_{num_samples}_chair.json",
        )
        with open(formulated_output_path, "w") as f:
            json.dump(formulated_output_dict, f)
        if verbosity:
            print(
                f"\nFormulated output matching CHAIR input format saved to {output_dir}."
            )


if __name__ == "__main__":

    # program level args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--decoder",
        type=str,
        default="greedy",
        help="Decoding strategy to use. You can choose from 'greedy', 'dola', 'halc'. Default is 'greedy'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="minigpt4",
        help="Name of the model. Default is 'minigpt4'.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coco",
        help="Name of the dataset. Default is 'coco'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/zhuokai/SN850X_4TB/Data/coco/val2014",
        help="Test data directory. Default is '/media/zhuokai/SN850X_4TB/Data/coco/val2014'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_chair_inputs/",
        help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of evaluation samples from the dataset. Default is 2000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Set universal seed. Default is 1.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_true",
        dest="verbosity",
        default=False,
        help="Verbosity. Default: False.",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=0,
        help="specify the gpu to load the model.",
    )

        # model specific parser
    parser_group = parser.add_argument_group("MiniGPT4")
    parser_group.add_argument(
        "--cfg_path",
        default="./eval_configs/minigpt4_llama2_eval_hallucination.yaml",
        help="path to configuration file.",
    )
    parser_group.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg_options instead.",
    )

    # load program level arguments
    args = parser.parse_args()

        # context_window = self.halc_params["context_window"]
        # # expand_ratio = 0.1
        # expand_ratio = self.halc_params["expand_ratio"]
    ###### PARAMETER SEARCH ######

    halc_params = {"context_domain":"upper"}

    for contrast_weight in [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]:
        for  context_window in [2, 3, 4, 5, 6]:
            for expand_ratio in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                halc_params["contrast_weight"] = contrast_weight
                halc_params["context_window"] = context_window
                halc_params["expand_ratio"] = expand_ratio
                hyper_params = {"halc_params": halc_params}
                print("hyper_params: ", hyper_params)
                generate_chair_input(hyper_params, args)
