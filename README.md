
# HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding

<div align="center">
  <img src="https://github.com/BillChan226/HALC/blob/obsolete/.assets/hawk_logo.png" width="35%">
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/pdf/2403.00425.pdf)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://billchan226.github.io/HALC.html)
[![GitHub Stars](https://img.shields.io/github/stars/BillChan226/HALC?style=social)](https://github.com/BillChan226/HALC/stargazers)

This repository provides the official PyTorch implementation of the following paper:
> [**HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding**]() <br>
> [Zhaorun Chen](https://billchan226.github.io/)<sup>1,\*</sup>,
> [Zhuokai Zhao](https://zhuokai-zhao.com/)<sup>1,\*</sup>,
> [Hongyin Luo](https://luohongyin.github.io/) <sup>2</sup>,
> [Huaxiu Yao](https://www.huaxiuyao.io/) <sup>3</sup>,
> [Bo Li](https://aisecure.github.io/)<sup>1,4</sup>,
> [Jiawei Zhou](https://sites.harvard.edu/jzhou/)<sup>5</sup>
>
> <sup>1</sup>University of Chicago, <sup>2</sup>Massachusetts Institute of Technology, <sup>3</sup>UNC-Chapel Hill <br> <sup>4</sup>University of Illinois at Urbana-Champaign, <sup>5</sup>Toyota Technological Institute at Chicago <br>
> <sub>*</sup> Equal contribution

## :partying_face: Features

:fire: We are actively enlarging HALC repo to support more VLMs & decoders. Stay tuned! If you wish to add some algorithm/model or contribute, please feel free to reach out to us!

#### Currently supported online OH decoding methods

|  Decoder |  [Minigpt4-v2](https://arxiv.org/abs/2304.10592) | [Instructblip](https://arxiv.org/abs/2305.06500) | [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | [mPLUG-OWL2](https://arxiv.org/abs/2311.04257) |
|----------|-----------|-------------|-----|-----|
| [Greedy*](https://github.com/huggingface/transformers)    | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [HALC*](https://arxiv.org/pdf/2403.00425.pdf)   | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [OPERA-Beam*](https://arxiv.org/abs/2311.17911)     | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [VCD](https://arxiv.org/abs/2311.16922)      | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [DoLa](https://arxiv.org/abs/2309.03883)*     | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

*: indicates the method supports beam search.

#### Currently supported post-hoc methods

|  Post-hoc |  [Minigpt4-v2](https://arxiv.org/abs/2304.10592) | [Instructblip](https://arxiv.org/abs/2305.06500) | [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | [mPLUG-OWL2](https://arxiv.org/abs/2311.04257) |
|----------|-----------|-------------|-----|-----|
| [Woodpecker](https://arxiv.org/abs/2310.16045)      | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [LURE](https://arxiv.org/abs/2310.00754)      | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## :hammer_and_wrench: Installation

To install, run the following commands to install the required packages:

```
git clone https://github.com/BillChan226/HALC.git
cd HALC
conda env create -f environment.yml
conda activate halc
```

We employ [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) as the external detector to ground hallucinatory objects. To install GroundingDINO with CUDA, we simplify the installation process, where you can:

```
# set CUDA_HOME to the virtual environment halc
export CUDA_HOME=$CONDA_PREFIX
# install GroundingDINO
cd decoder_zoo/GroundingDINO
pip install -e .
# go back to HALC root
cd ../..
```

To download pre-trained model weights for DINO:

```
# default directory that contains the weights
mkdir model_checkpoints
cd model_checkpoints
# download weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# go back to HALC root
cd ..
```

## :bee: LVLM Backbones

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

Besides, you need to prepare the following checkpoints of 7B base models:

- Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 14](https://github.com/BillChan226/HALC/blob/924cdc09310df8826fe2f8e2e16c25a6312a48b7/eval_configs/llava-1.5_eval.yaml#L14) of `eval_configs/llava-1.5_eval.yaml`.
- Download [LLaMA-2 7B model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) and specify it at [Line 15](https://github.com/BillChan226/HALC/blob/924cdc09310df8826fe2f8e2e16c25a6312a48b7/minigpt4/configs/models/minigpt4_llama2.yaml#L15) of `minigpt4/configs/models/minigpt4_llama2.yaml`.
- Download [Vicuna 7B v1.1 model](https://github.com/lm-sys/FastChat) and specify it at [Line 25](https://github.com/BillChan226/HALC/blob/924cdc09310df8826fe2f8e2e16c25a6312a48b7/minigpt4/configs/models/blip2_instruct_vicuna7b.yaml#L25) of `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml`.
- Download [Vicuna 7B v0 model](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it at [Line 18](https://github.com/BillChan226/HALC/blob/924cdc09310df8826fe2f8e2e16c25a6312a48b7/minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) of `minigpt4/configs/models/minigpt4_vicuna0.yaml`.
- Download [MiniGPT-4 7B pretrained weights](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and specify it at [Line 8](https://github.com/BillChan226/HALC/blob/924cdc09310df8826fe2f8e2e16c25a6312a48b7/eval_configs/minigpt4_eval.yaml#L8C10-L8C10) of `eval_configs/minigpt4_eval.yaml`.
- Download [MiniGPT-4 7B pretrained weights for LlaMA-2](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and specify it at [Line 8](https://github.com/BillChan226/HALC/blob/924cdc09310df8826fe2f8e2e16c25a6312a48b7/eval_configs/minigpt4_llama2_eval.yaml#L8) of `eval_configs/minigpt4_llama2_eval.yaml`.
- Download [mPLUG-Owl2 7B pretrained weights](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b) and specify it at [Line 14](https://github.com/BillChan226/HALC/blob/924cdc09310df8826fe2f8e2e16c25a6312a48b7/eval_configs/mplug-owl2_eval.yaml#L14) of `eval_configs/mplug-owl2_eval.yaml`.

### Arguments

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--model`    | `llava-1.5` | Specify the MLLM model, this codebase supports `instructblip`, `minigpt4`, `llava-1.5`. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder, e.g., `COCO_2014/val2014/`. |
| `--pope-type`     | `random` | Type for POPE evaluation, supports `random`, `popular`, `adversarial`. |
| `--beam`   | `3` | Beam size for global search. Default: 1. |

#### Arguments for HALC

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--k-candidate-num`      | `4` | Number of generative focal fields for local search. Default: 4. |
| `--expand-ratio`      | `0.6` | The growing factor of focal fields. Default: 0.6. |
| `--detector`      | `dino` | Detector to use in [dino, owlv2]. Default: dino. |
| `--box_threshold`      | `0.4` | The threshold for bounding box in GroundingDino. Default: 0.4. |

#### Arguments for OPERA

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--scale_factor`   | `50` | The scale factor to scale up the self-attention weights. Default: 50. |
| `--threshold`      | `15` | The threshold for attending retrospection. Default: 15. |
| `--num_attn_candidates`   | `5` | The number of candidates per beam. Default: 5. |
| `--penalty_weights`| `1` | The weight of penalty term in decoding. Default: 1.  |

#### Arguments for VCD

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--cd-alpha`      | `1` | Amplification factor. Default: 1. |
| `--cd-beta`   | `0.1` | Truncation factor for adaptive plausibility constraint. Default: 0.1. |
| `--noise-step`| `500` | Number of steps to add diffusion noise. Default: 500.  |

## :hourglass: Benchmarking OH

#### :chair: Running CHAIR evaluation for LVLMs object hallucination

Following [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/pdf/2305.10355.pdf), we used "Please describe this image in detail." as the prompt to query LVLM for captions of the `500` images randomly sampled from [COCO 2014 Val](https://cocodataset.org/#download) datast. Under root directory, run

```
python run_scripts/caption_generation.py --model [LVLM Backbone] --data_path [COCO_DIR] -d [Decoding Strategy] --num_samples 500 --seed [SEED] --gpu-id [GPU_IDs] --output_dir ./generated_captions/ --debugging 1
```

`--debugging 1` will print the intermediate hallucination correction process of HALC.

#### :man_in_tuxedo: Running POPE evaluation for LVLMs object hallucination

Since OPOPE evaluates directly based on the caption generated for each image, it follows the caption generation procedure for CHAIR and differs in the subsequent metric calculation. To collect samples for the conventional POPE evaluation, under root directory, run

```
python run_scripts/pope_eval.py --model [LVLM Backbone] --data_path [COCO_DIR] -d [Decoding Strategy] --pope_type [random/popular/adversarial] --num_images 100 --seed [SEED] --gpu_id [GPU_IDs] --output_dir ./generated_captions/
```

#### :woman_juggling: Running MME Benchmark to evaluate LVLMs object hallucination

MME also follows the same procedure as CHAIR and OPOPE to collect samples. Alternatively, under root directory, run

```
python run_scripts/mme_eval.py --model [LVLM Backbone] --data_path [MME_DIR] -d [Decoding Strategy] --num_samples 30 --seed [SEED] --gpu-id [GPU_IDs] --output_dir ./generated_captions/
```


#### :hospital: Running post-hoc methods to revise generated captions

Under root directory, run

```
python run_scripts/reviser_eval.py -r [woodpecker/lure] --data_path [COCO_DIR] --c [PATH_TO_CAPTION] --seed [SEED] --gpu-id [GPU_IDs] --output_dir ./log/
```

### Evaluation

#### CHAIR Scores

After preparing your caption files using the above commands, you can either choose to evaluate the captions in an **one-shot mode** (single caption) or **batch mode** (all the caption files in a folder). To evaluate a single caption file,

```
python eval/eval_hallucination.py --metric chair --chair_input_path [PATH_TO_CAPTION_DIR] -v
```

To evaluate a batch of caption files, run

```
python eval/caption_to_chair.py -c [PATH_TO_CAPTION_FOLDER_DIR]
```

to convert the caption files to the format ready for CHAIR evaluation in the same directory first. Then a `_chair.json` file will be produced under this folder. To further evaluate the CHAIR score as well as the generation quality scores, run
```shell
python eval/batch_eval.py -c [PATH_TO_CAPTION_FOLDER_DIR] --evaluator chair --coco_path [COCO_DIR]
```

Note that `[COCO_DIR]` is expected to contain both images and annotation files within the `annotations` subfolder. In other words, `[COCO_DIR]` should the the following structure:

```
COCO_DIR (val2014 for example)
  - annotations
    - captions_val2014.json
    - captions_val2014.json
    - instances_train2014.json
    - instances_val2014.json
    - person_keypoints_train2014.json
    - person_keypoints_val2014.json
  - COCO_val2014_000000000042.jpg
  - COCO_val2014_000000000073.jpg
  ...
```

#### POPE Scores

Similarly, you can also evaluate POPE in both modes. To evaluate a single caption file,
```shell
python eval_hallucination.py --metric pope --pope_answer_path [PATH_TO_CAPTION_DIR] --pope_question_path [PATH_TO_POPE_QUESTION] -v
```

To evaluate a batch of caption files, run
```shell
python eval/batch_eval.py -c [PATH_TO_CAPTION_FOLDER_DIR] --evaluator pope --pope_type [random/popular/adversarial]
```

The evaluation results will be saved in the same directory.


#### MME Scores


To evaluate the MME scores on each chosen subset, modify the `subset_dir` variable [here](https://github.com/BillChan226/HALC/blob/fc32840be64a4f63863a87fbc1beedef47805955/eval/MME_score.py#L195) to include the list of directories of your target directories and run
```shell
python eval/MME_eval.py
```


<details>
  <summary>Click to view how to run some interesting demo!</summary>

  ## :roller_coaster: Demo Playgrounds

  ### :eagle: HALC Demo

  Run CDL demo on a [toy example](hallucinatory_image/beach_on_a_clock.png):

  ```
  python context_density/context_decoding.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
  ```

  ### ViT Early Exit Layers Demo

  Specify early_exit_layer_idx then run ViT early exit layers contrastive decoding:

  ```
  python vit_early_exit_contrast.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
  ```

  ### DoLa Demo

  #### Test DoLa with their textual input

  run

  ```
  python toy_dola_eval.py --model-name ./models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8 --output-path output-path.json --num-gpus 1 --early-exit-layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32
  ```

  Note: adding 32 in the early-exit-layers is crucial for reasonable output.

  JSD for each candidate layer is printed and input at line 2720 of file ```DoLa/transformers-4.28.1/src/transformers/generation/utils.py```

  #### Test DoLA with visual-textual input

  run a toy example:

  ```
  python contrast_decoding.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
  ```

  The [toy example](hallucinatory_image/beach_on_a_clock.png) is projected into the prefix of the language model as a context.

</details>

## :wrench: Troubleshooting

#### Debugging HALC

Add `--debugger` argument to the command line to print the debugging output for HALC. Specifically, use `--debugger 1` to print the intermediate hallucination correction with HALC, and `--debugger 2` to print all the debugging output. For example:
```
python run_scripts/caption_generation.py --debugger 1
```


#### Error installing `GroundingDINO`

If error `NameError: name '_C' is not defined` is reported, refer to [this issue](https://github.com/IDEA-Research/GroundingDINO/issues/8#issuecomment-1541892708) for a quick fix.

#### Error installing `pattern`

```
conda install -c conda-forge pattern
```

#### CUDA Error installing `GroundingDINO`

```
conda install pytorch torchvision torchaudio pytorch-cuda=[YOUR NVIDIA CUDA VERSION] -c pytorch -c nvidia
```

#### runtimeError: Input type (float) and bias type (c10::Half) should be the same

simply reinstall torch==2.0.0 will most likely solve the issue

```
pip uninstall torch
pip install torch==2.0.0
```

## :book: Acknowledgement
Please cite the paper as follows if you use the data or code from HALC:
```
@article{chen2024halc,
  title={HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding},
  author={Chen, Zhaorun and Zhao, Zhuokai and Luo, Hongyin and Yao, Huaxiu and Li, Bo and Zhou, Jiawei},
  journal={arXiv preprint arXiv:2403.00425},
  year={2024}
}
```

## :book: Contact
Please reach out to us if you have any suggestions or need any help in reproducing the results. You can submit an issue or pull request, or send an email to zhaorun@uchicago.edu.

## :key: License

This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with
BSD 3-Clause License [here](LICENSE_Lavis.md).
