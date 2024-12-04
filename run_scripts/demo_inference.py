import argparse
import os
import random
import sys
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json

from decoder_zoo.Woodpecker.vis_corrector import Corrector
from decoder_zoo.HALC.context_density.halc import halc_assistant
from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "eval_configs/mplug-owl2_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:"
}


def setup_seeds(config, seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="minigpt4", help="model")
parser.add_argument(
    "-d",
    "--decoder",
    type=str,
    default="greedy",
    help="Decoding strategy to use. You can choose from 'greedy', 'dola', 'halc'. Default is 'greedy'.",
)
parser.add_argument(
    "-g", "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
)
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")
parser.add_argument("-b", "--beam", type=int)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "-v",
    "--verbosity",
    action="store_false",
    dest="verbosity",
    default=True,
    help="Verbosity. Default: True.",
)
parser.add_argument(
    "-k",
    "--k-candidate-num",
    type=int,
    default=2,
    help="specify the k candidate number for halc.",
)
parser.add_argument(
    "-p",
    "--post-correction",
    type=str,
    default=None,
    help="Post correction method such as Woodpecker, Lure.",
)
parser.add_argument(
    "--cd_alpha",
    type=float,
    default=1,
    help="Alpha param for VCD.",
)
parser.add_argument("--cd_beta", type=float, default=0.1, help="Beta param for VCD.")
parser.add_argument("--noise_step", type=int, default=3, help="Noise step for VCD.")
parser.add_argument(
    "--detector",
    type=str,
    default="dino",
    help="Detector type. Default is 'groundingdino'.",
)
parser.add_argument("--box_threshold", type=float, default=0.08, help="Box threshold for DINO.")
parser.add_argument(
    "-e",
    "--expand-ratio",
    type=float,
    default=0.6,
    help="Expand ratio of growing contextual field.",
)
parser.add_argument(
    "--debugger",
    action="store_true",
    default=False,
    help="Whether to use debugger output.",
)

args = parser.parse_known_args()[0]

print("args.gpu_id", args.gpu_id)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
model_name = args.model
decoding_strategy = args.decoder
cfg = Config(args)
seed = args.seed
setup_seeds(cfg, seed)

device = (
    torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
)

verbosity = args.verbosity
k_candidate_num = args.k_candidate_num
num_beams = args.beam
num_workers = args.num_workers
batch_size = args.batch_size
post_correction = args.post_correction
cd_alpha = args.cd_alpha
cd_beta = args.cd_beta
detector_type = args.detector
expand_ratio = args.expand_ratio
debugger = args.debugger
box_threshold = args.box_threshold

# ========================================
#             Model Initialization
# ========================================
print("Initializing Model")

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)


valid_decoding_strategies = ["greedy", "dola", "halc-beam", "opera-beam", "vcd"]

assert (
    decoding_strategy in valid_decoding_strategies
), f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}"

decoding_strategy = decoding_strategy
opera_decoding = False
dola_decoding = False
halc_decoding = False
beam_search = False
vcd_decoding = False

if decoding_strategy == "greedy":
    pass
elif decoding_strategy == "dola":
    dola_decoding = True

elif decoding_strategy == "halc-dola":
    dola_decoding = True
    halc_decoding = True
elif decoding_strategy == "halc-greedy":
    halc_decoding = True
elif decoding_strategy == "halc-beam":
    halc_decoding = True
    dola_decoding = True
    beam_search = True
elif decoding_strategy == "opera-beam":
    beam_search = True
    opera_decoding = True
elif decoding_strategy == "vcd":
    vcd_decoding = True


if post_correction == "woodpecker":
    model_args = SimpleNamespace(**args_dict)
    corrector = Corrector(model_args)


print(f"\033[42m####### Current Decoding Strategy: {decoding_strategy} #######\033[0m")


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)


if verbosity:
    print("\ndecoding strategy: ", decoding_strategy)
    print("backbone model_name: ", args.model)
    print("num_beams: ", num_beams)
    print("seed: ", seed)
    print(vis_processors["eval"].transform)



# image_path = "/home/czr/HaLC/hallucinatory_image/beach_on_a_clock.png"
# image_path = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/test.png"
# image_path = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/zoom_in_5.png"
# image_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000070294.jpg"
# image_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000332775.jpg"
# image_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000350132.jpg"
# image_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000302222.jpg" 
# image_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000444366.jpg" 
# image_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000085926.jpg" 
# image_path = "/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/COCO_val2014_000000165257.jpg" # 162543
# image_path = "/home/czr/HaLC/LLaVA-Bench/001.jpg"
img_path_dir = "/home/czr/MMHallucination/co-occurrence/dataset/image_to_text/high_cooc/object_existence/optimization/"
img_path_list = os.listdir(img_path_dir)
# generated_captions_path = f"llava-bench/{decoding_strategy}_llava-1.5.json"

img_path_list = ["/home/czr/MMHallucination/co-occurrence/dataset/image_to_text/high_cooc/object_existence/normal/person_surfboard/COCO_val2014_000000018358.jpg_inpainted.jpg", 
                 "/home/czr/MMHallucination/co-occurrence/dataset/image_to_text/high_cooc/attribute/normal/pepper_red/COCO_val2014_000000243091.jpg_inpainted_pepper.jpg"]

print("img_path_list", img_path_list)
for image_path in img_path_list:
    img_save = {}
    img_save["image_path"] = image_path
    # image_path = img_path_dir + image_path
    raw_image = Image.open(image_path).convert("RGB")

    if model_name == "mplug-owl2":
        max_edge = max(raw_image.size) # We recommand you to resize to squared image for BEST performance.
        image = raw_image.resize((max_edge, max_edge))
        image_tensor = process_images([image], model.image_processor)
        image = image_tensor.to(device, dtype=torch.float16)
    else:
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device)

    qu = "Please describe this image in detail."
    # qu = "Generate a one sentence caption of the image."
    # qu = "Generate a short caption of the image."
    # qu = "What is the man holding in his hand?"
    # qu = "generate a one sentence caption of the image"


    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)


    # halc_params = {"context_domain": "upper", "contrast_weight": 0.05, "context_window": 4, "expand_ratio": 0.1, "beam_size": num_beams, "k_candidate_num": args.k_candidate_num, "LVLM_backbone": model_name, "detector": "groundingdino"}
    halc_params = {
        "context_domain": "upper",
        "contrast_weight": 0.05,
        "context_window": 3,
        "expand_ratio": expand_ratio,
        "beam_size": num_beams,
        "k_candidate_num": args.k_candidate_num,
        "LVLM_backbone": model_name,
        "detector": detector_type,
        "score_type": "BLIP",
        "debugger": debugger,
        "box_threshold": box_threshold,
    }

    halc_assistant_helper = halc_assistant(
        model, vis_processor=vis_processor, device=device, halc_params=halc_params
    )

    lm_early_exit_layers = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
    ]

    mature_layer = lm_early_exit_layers[-1]
    premature_layer = None
    candidate_premature_layers = lm_early_exit_layers[:-1]
    premature_layer_dist = {l: 0 for l in candidate_premature_layers}

    halc_assistant_helper.update_input(img_path=image_path, input_prompt=qu)


    image_cd = None
    if decoding_strategy == "vcd":
        image_tensor_cd = add_diffusion_noise(image, args.noise_step)
        image_cd = image_tensor_cd.unsqueeze(0).half().to(device) if image_tensor_cd is not None else None
        cd_alpha = cd_alpha
        cd_beta = cd_beta
        if model_name == "minigpt4":
            image_cd = image_cd.squeeze(0)


    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt":qu, "img_path": image_path},
                use_nucleus_sampling=args.sample, 
                num_beams=num_beams,
                max_new_tokens=128,
                output_attentions=True,
                premature_layer=premature_layer,
                candidate_premature_layers=candidate_premature_layers,
                mature_layer=mature_layer,
                beam_search=beam_search,
                dola_decoding=dola_decoding,
                opera_decoding=opera_decoding,
                vcd_decoding=vcd_decoding,
                halc_decoding=halc_decoding,
                # HALC
                halc_assistant=halc_assistant_helper,
                # OPERA
                key_position=None,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                # VCD
                images_cd=image_cd,
                cd_alpha=cd_alpha,
                cd_beta=cd_beta,
            )

    output_text = out[0]

    print("original output text", output_text)
    # sentence_list = output_text.replace('.', ',').split(',')
    # sentence_filter_list = []
    # for sentence in sentence_list:
    #     if "unk" not in sentence:
    #         sentence_filter_list.append(sentence)
    # output_text = ",".join(sentence_filter_list)

    # print("decoder output text", output_text)

    if post_correction == "woodpecker":
        sample = {
            "img_path": image_path,
            "input_desc": output_text,
            "query": "Generate a short caption of the image.",
        }

        corrected_sample = corrector.correct(sample)
        output_text = corrected_sample["output"]
        print("corrected output_text", output_text)


    print("image_path", image_path)
    print("caption: ", output_text)
    img_save["caption"] = output_text

    input()

    # with open(generated_captions_path, "a") as f:
    #     json.dump(img_save, f)
    #     f.write("\n")

