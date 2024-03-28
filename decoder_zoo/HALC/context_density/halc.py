import random
import numpy as np
import torch
import json
from decoder_zoo.HALC.context_density.detector import Detector
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace
from PIL import Image, ImageDraw
import spacy
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipModel
import random
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from PIL import Image, ImageFilter
import hpsv2
from mplug_owl2.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

exempt_word_list = ["image", "side", "background", "feature", "features", "center", 
                    "left", "right", "scene", "view", "s", "Birthday", "detail", "red",
                    "white", "cat", "horse", "bus", "group", "manner", "her", "birds", 
                    "teddy", "stack", "cell", "toaster", "mirror", "toilet", "captures",
                    "boat", "bench"]

add_word_list = ["sink", "microwave", "toaster", "puppy", "bottle", "table", "oven", 
                "orange", "toothbrush", "cars"]

class halc_assistant:
    def __init__(
        self,
        model=None,
        vis_processor=None,
        device=None,
        halc_params=None,
        max_new_tokens=64,
    ):
        # initialize detector
        args_dict = {
            "detector_config": "decoder_zoo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "detector_model_path": "decoder_zoo/GroundingDINO/weights/groundingdino_swint_ogc.pth",
            "cache_dir": "decoder_zoo/HALC/cache_dir",
            "device": device,
        }
        model_args = SimpleNamespace(**args_dict)
        self.device = device
        self.debugger = halc_params["debugger"]
        halc_detector = halc_params["detector"]
        if halc_detector == "dino":
            self.detector = Detector(model_args, self.debugger)
        elif halc_detector == "owlv2":
            self.owlv2_processor = Owlv2Processor.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
        else:
            raise ValueError("Invalid detector!")

        self.vis_processor = vis_processor
        self.model = model

        self.box_threshold = halc_params["box_threshold"]
        self.tagging = spacy.load("en_core_web_sm")
        # self.tagging_sm = spacy.load("en_core_web_sm")
        # self.tagging_md = spacy.load("en_core_web_md")
        # self.tagging_lg = spacy.load("en_core_web_md")
        self.halc_params = halc_params
        self.k_candidate_num = halc_params["k_candidate_num"]
        # self.original_image = None
        self.grounded_check = False
        self.max_new_tokens = max_new_tokens

        self.max_handle_box = 3
        self.skip_rate = 0.3

        self.exempt_word_list = exempt_word_list
        self.add_word_list = add_word_list

        self.model_backbone = halc_params["LVLM_backbone"]

        if self.model_backbone == "minigpt4" or self.model_backbone == "llava-1.5":
            self.tokenizer = self.model.llama_tokenizer
            token_vocab_dir = "decoder_zoo/HALC/context_density/llama_tokenizer.json"
        elif self.model_backbone == "instructblip":
            self.tokenizer = self.model.llm_tokenizer
            token_vocab_dir = "decoder_zoo/HALC/context_density/vicuna_tokenizer.json"
        elif self.model_backbone == "mplug-owl2":
            self.tokenizer = self.model.llm_tokenizer
            token_vocab_dir = "decoder_zoo/HALC/context_density/llama_tokenizer.json"

        with open(token_vocab_dir, "r") as f:
            self.token_vocab = json.load(f)

        self.token_vocab = self.token_vocab["model"]["vocab"]

        self.token_vocab = {value: key for key, value in self.token_vocab.items()}

        score_type = halc_params["score_type"]
        if score_type == "CLIP":
            self.score_type = "CLIP"
            if self.max_new_tokens > 77:
                config_text = CLIPTextConfig(max_position_embeddings=self.max_new_tokens)
                config_vision = CLIPVisionConfig()
                config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
                self.score_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    config=config,
                    ignore_mismatched_sizes=True,
                )
                self.score_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    config=config,
                    ignore_mismatched_sizes=True,
                )
            else:
                self.score_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.score_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
        elif score_type == "BLIP":
            self.score_type = "BLIP"
            self.score_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large")
            self.score_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
        elif score_type == "Random":
            self.score_type = "Random"
            self.score_model = None
            self.score_processor = None
        elif score_type == "Perplexity":
            self.score_type = "Perplexity"
            self.score_processor = None
            self.score_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            self.score_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map="auto")
        elif score_type == "HPSv2":
            self.score_type = "HPSv2"
            self.score_model = hpsv2
            self.score_processor = None
        else:
            raise ValueError("Invalid score type!")


    def calculate_perplexity(self, text):
        input_ids = torch.tensor(self.score_tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.score_model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return torch.exp(loss).item()

    def update_input(self, img_path, input_prompt):
        # print("img_path", img_path)
        self.detector_dict = {"img_path": img_path}
        self.image_to_ground = Image.open(img_path)
        self.prompt = input_prompt
        self.original_image = img_path

    def reset_info(self):
        self.detector_dict = None
        self.image_to_ground = None
        self.prompt = None
        self.original_image = None
        self.grounded_check = False

    def check_word_complete(self, input_id):
        if isinstance(input_id, torch.Tensor):
            input_id = input_id.cpu().numpy().tolist()
            
        # print("input_id", input_id)
        # if input_id[0][0] == -1:
        #     return True
        final_tokens = self.token_vocab[input_id[0][0]]
        # print("final_tokens", final_tokens)
        if "▁" in final_tokens or "." in final_tokens or input_id[0][0] == 2:
            last_word_flag = True
        else:
            last_word_flag = False

        return last_word_flag

    def get_last_word(self, input_ids):
        output_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        last_word_flag = True
        last_word = output_text.split(" ")[-1]

        return last_word

    def get_sequence_text(self, input_ids, skip_token_length=None):
        if skip_token_length == 0 or skip_token_length is None:
            output_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(
                input_ids[skip_token_length:], skip_special_tokens=True
            )

        return output_text

    def compute_bbox_size(self, bbox):
        """
        Computes the size of a bounding box.
        """
        # bbox = [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        return width * height

    def expand_bbox(self, bbox, context_expansion_factor):
        """
        Expands the bounding box by a given expansion factor.
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        expanded_x_min = max(0, x_min - width * context_expansion_factor / 2)
        expanded_y_min = max(0, y_min - height * context_expansion_factor / 2)
        expanded_x_max = min(1, x_max + width * context_expansion_factor / 2)
        expanded_y_max = min(1, y_max + height * context_expansion_factor / 2)
        

        return [expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max]

    def draw_bbox(self, image, bbox, color="yellow", width=3):
        """
        Draws a bounding box on an image.

        :param image: The image on which to draw.
        :param bbox: The bounding box coordinates as a list of [x_min, y_min, x_max, y_max].
        :param color: The color of the box.
        :param width: The line width of the box.
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        # Convert normalized bbox coordinates to absolute pixel values
        rect = [
            bbox[0] * im_width,
            bbox[1] * im_height,
            bbox[2] * im_width,
            bbox[3] * im_height,
        ]
        # Draw the rectangle on the image
        draw.rectangle(rect, outline=color, width=width)
        return image

    def compute_jsd(self, p, q):
        # calculate the kl divergence
        def kl_divergence(p, q):
            return sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)))

        # calculate the js divergence
        m = 0.5 * (p + q)
        jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        print(jsd)
        exit()

        return jsd

    def context_density_embedding(self, entity):
        # context_window specifies the number of context windows

        context_window = self.halc_params["context_window"]
        expand_ratio = self.halc_params["expand_ratio"]

        entity = entity.strip(".").strip(",").strip("'").strip("]").strip("[").strip(")")
        if len(entity) > 0:
            if entity[-1] == "s":
                entity = entity[:-1]

        doc = self.tagging(entity)
        # doc_sm = self.tagging_sm(entity)
        # doc_md = self.tagging_md(entity)
        # doc_lg = self.tagging_lg(entity)
        detect_info = {}

        # print("entity", entity)
        if len(doc) < 1:
            detect_info["pos"] = "PUNC"
            # detect_info["pos_sm"] = "PUNC"
            # detect_info["pos_md"] = "PUNC"
            # detect_info["pos_lg"] = "PUNC"
        else:
            detect_info["pos"] = doc[0].pos_

            # detect_info["pos_sm"] = doc_sm[0].pos_
            # detect_info["pos_md"] = doc_md[0].pos_
            # detect_info["pos_lg"] = doc_lg[0].pos_
            
        # add a random filter to halc verification

        
        if "." in entity or "," in entity:
            detect_info["pos"] = "SKIP"
        if entity in self.exempt_word_list:
            detect_info["pos"] = "SKIP"
        if random.random() < self.skip_rate:
            detect_info["pos"] = "SKIP"
        #     detect_info["pos_sm"] = "SKIP"
        #     detect_info["pos_md"] = "SKIP"
        #     detect_info["pos_lg"] = "SKIP"

        if entity in self.add_word_list:
            detect_info["pos"] = "ADD"
            # detect_info["pos_sm"] = "ADD"
        #     detect_info["pos_md"] = "ADD"
        #     detect_info["pos_lg"] = "ADD"


        # print("ENTITY: ", entity)
        # print("pos", detect_info["pos"])

        valid_list = ["NOUN", "PROPN", "ADD"] #, "ADJ"]

        # if detect_info["pos_sm"] in valid_list or detect_info["pos_md"] in valid_list or detect_info["pos_lg"] in valid_list:
        if detect_info["pos"] in valid_list:
            detect_info["status"] = "activated"
            self.detector_dict["named_entity"] = [entity]
            self.detector_dict["box_threshold"] = self.box_threshold

            if self.halc_params["detector"] == "dino":
                sample = self.detector.detect_objects(self.detector_dict)

                if self.debugger == 2:
                    print("Detection: ", sample)
                # Assuming the first detected bounding box is the one related to the entity
                try:
                    original_bbox = sample["entity_info"][entity]["bbox"]
                except:
                    original_bbox = []

                # print("original_bbox", original_bbox)

            elif self.halc_params["detector"] == "owlv2":
                # print("entity", entity)
                img_path = self.detector_dict["img_path"]
                # image_to_ground = Image.open(img_path)
                entity_to_ground = [self.detector_dict["named_entity"]]
                # entity_to_ground = [["a man hold a clock"]]
                # print("texts", entity_to_ground)
                # print("self.detector_dict", self.detector_dict)
                owlv2_inputs = self.owlv2_processor(
                    text=entity_to_ground,
                    images=self.image_to_ground,
                    return_tensors="pt",
                )
                owlv2_outputs = self.owlv2_model(**owlv2_inputs)
                # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                # target_sizes = torch.Tensor([self.image_to_ground.size[::-1]])
                # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
                # results = self.owlv2_processor.post_process_object_detection(outputs=owlv2_outputs, target_sizes=target_sizes, threshold=0.1)

                results = self.owlv2_processor.post_process_object_detection(
                    outputs=owlv2_outputs, threshold=0.05
                )

                # print("results: ", results)
                original_bbox = results[0]["boxes"].cpu().numpy().tolist()

            if len(original_bbox) > self.max_handle_box:
                detect_info["status"] = "invalid"
                embeds_list = None
                return embeds_list, detect_info

            if len(original_bbox) == 0:
                # target_bbox = [0.3, 0.3, 0.6, 0.6]
                target_bbox = [0.2, 0.2, 0.8, 0.8]
                detect_info["status"] = "not-detected"
                # detect_info["status"] = "invalid"
                # embeds_list = None
                # return embeds_list, detect_info
            else:
                area_list = [
                    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in original_bbox
                ]

                # get the index of the smallest bbox
                target_bbox_index = area_list.index(min(area_list))

                # target_bbox_index = area_list.index(max(area_list))

                # target_bbox_index = sorted(
                #     range(len(area_list)), key=lambda k: area_list[k]
                # )[len(area_list) // 2]

                target_bbox = original_bbox[target_bbox_index]

            # target_bbox = original_bbox[0]

            # Calculate expanded bounding boxes for the given context window
            if context_window == 1:  # only the original one
               
                expanded_bboxes = [target_bbox]
            else:
                # check on the target box's size
                target_bbox_size = self.compute_bbox_size(target_bbox)
                # smallest size should be 0.2

                if expand_ratio > 0.8:
                    expanded_bboxes = [
                        self.expand_bbox(target_bbox, -0.8),
                        target_bbox,
                    ]
                else:
                    expanded_bboxes = [
                        self.expand_bbox(target_bbox, -expand_ratio),
                        target_bbox,
                    ]
                all_box_sizes = [
                    self.compute_bbox_size(expanded_bboxes[0]),
                    self.compute_bbox_size(expanded_bboxes[1]),
                ]

                for _ in range(1, context_window - 1):
                    # Each expansion is double the size of the previous level
                    expanded_bboxes.append(
                        self.expand_bbox(expanded_bboxes[-1], expand_ratio)
                    )
                    all_box_sizes.append(
                        self.compute_bbox_size(expanded_bboxes[-1])
                    )

                # index of the original target box
                self.target_bbox_index = min(
                    range(len(all_box_sizes)),
                    key=lambda i: abs(all_box_sizes[i] - target_bbox_size),
                )


            self.grounded_check = True

            original_image = self.image_to_ground.convert("RGB")

            # Crop images to the expanded bounding boxes
            cropped_images = []
            for bbox in expanded_bboxes:
                # print("bbox", bbox)
                # Calculate the absolute coordinates of the bounding box
                im_width, im_height = original_image.size
                left = bbox[0] * im_width
                top = bbox[1] * im_height
                right = bbox[2] * im_width
                bottom = bbox[3] * im_height

                # Crop the image to the bounding box
                cropped_image = original_image.crop((left, top, right, bottom))
                cropped_images.append(cropped_image)

            # cropped_images[1].save(f"/home/czr/HaLC/decoder_zoo/HALC/cache_image/cropped_{entity}.png")
            # print(f"/home/czr/HaLC/decoder_zoo/HALC/cache_image/cropped_{entity}.png")
            # Save the cropped images
            # saved_paths = []
            # for i, cropped_img in enumerate(cropped_images, start=1):
            #     save_path = f"/home/czr/HaLC/decoder_zoo/HALC/cache_dir/cropped_level_{i}.png"
            #     cropped_img.save(save_path)
            #     saved_paths.append(save_path)

            # input()
            # get decoding for each context window

            embeds_list = []
            for i, cropped_img in enumerate(cropped_images):
                embs = self.get_model_embeds(cropped_img)
                embeds_list.append(embs)
        else:
            detect_info["status"] = "invalid"
            embeds_list = None

        return embeds_list, detect_info

    def get_model_embeds(self, image):

        if self.model_backbone == "minigpt4":
            max_new_tokens = self.max_new_tokens
            max_length = 512

            image = self.vis_processor(image).unsqueeze(0).to(self.device)
            image_emb, _ = self.model.encode_img(image, 38)

            # prompt = self.prompt[0]
            prompt = self.prompt

            embs = self.model.get_context_emb(prompt, [image_emb])

            current_max_len = embs.shape[1] + max_new_tokens

            if current_max_len - max_length > 0:
                print(
                    "Warning: The number of tokens in current conversation exceeds the max length. "
                    "The model will not see the contexts outside the range."
                )

            begin_idx = max(0, current_max_len - max_length)
            embs = embs[:, begin_idx:]

        elif self.model_backbone == "llava-1.5":
            # image_emb = self.model.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, image)
            embs = self.vis_processor(image).unsqueeze(0).to(self.device)
        elif self.model_backbone == "mplug-owl2":
            # image_emb = self.model.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, image)
            max_edge = max(
                image.size
            )  # We recommand you to resize to squared image for BEST performance.
            image = image.resize((max_edge, max_edge))
            image_tensor = process_images([image], self.model.image_processor)
            embs = image_tensor.to(self.device, dtype=torch.float16)

        elif self.model_backbone == "instructblip":
            image = self.vis_processor(image).unsqueeze(0).to(self.device)
            image_emb, _ = self.model.encode_img(image, 38)
            embs = self.model.image_to_embs(image_emb, image)
        else:
            raise NotImplementedError

        return embs

    def context_density_distortion_embedding(self, entity):

        raise NotImplementedError
        # context_window specifies the number of context windows

        context_window = self.halc_params["context_window"]
        # expand_ratio = 0.1
        expand_ratio = self.halc_params["expand_ratio"]

        entity = entity.strip(".")
        doc = self.tagging(entity)
        detect_info = {}

        if len(doc) < 1:
            detect_info["pos"] = "PUNC"
        else:
            detect_info["pos"] = doc[0].pos_

        # print("entity", entity)
        # print("pos", detect_info["pos"])

        valid_list = ["NOUN", "PROPN"]

        if detect_info["pos"] in valid_list:
            detect_info["status"] = "activated"
            self.detector_dict["named_entity"] = [entity]
            sample = self.detector.detect_objects(self.detector_dict)
            
            # print("Detection: ", sample)
            # Assuming the first detected bounding box is the one related to the entity

            original_bbox = sample["entity_info"][entity]["bbox"]
            if len(original_bbox) > 2:
                detect_info["status"] = "invalid"
                embeds_list = None
                return embeds_list, detect_info

            if len(original_bbox) == 0:
                target_bbox = [0.3, 0.3, 0.6, 0.6]
                detect_info["status"] = "bounding box not detected"
            else:
                area_list = [
                    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in original_bbox
                ]

                # get the index of the smallest bbox
                target_bbox_index = area_list.index(min(area_list))
                target_bbox = original_bbox[target_bbox_index]

            # target_bbox = original_bbox[0]

            # Calculate expanded bounding boxes for the given context window
            if context_window == 1:
                # only the original one
                expanded_bboxes = [target_bbox]
            else:
                # check on the target box's size
                target_bbox_size = self.compute_bbox_size(target_bbox)
                # smallest size should be 0.2
                if target_bbox_size < 0.2:
                    # only increase the size
                    expanded_bboxes = [
                        self.expand_bbox(target_bbox, -expand_ratio),
                        target_bbox,
                    ]
                    for _ in range(1, context_window - 1):
                        # Each expansion is double the size of the previous level
                        expanded_bboxes.append(
                            self.expand_bbox(expanded_bboxes[-1], expand_ratio)
                        )

                    # index of the original target box
                    self.target_bbox_index = 0
                else:
                    initial_ratio = np.sqrt(0.2 / target_bbox_size)
                    # expanded_bboxes = [self.expand_bbox(target_bbox, -initial_ratio)]
                    expanded_bboxes = [
                        self.expand_bbox(target_bbox, -expand_ratio),
                        target_bbox,
                    ]
                    all_box_sizes = [
                        self.compute_bbox_size(expanded_bboxes[0]),
                        self.compute_bbox_size(expanded_bboxes[1]),
                    ]

                    for _ in range(1, context_window - 1):
                        # Each expansion is double the size of the previous level
                        expanded_bboxes.append(
                            self.expand_bbox(expanded_bboxes[-1], expand_ratio)
                        )
                        all_box_sizes.append(
                            self.compute_bbox_size(expanded_bboxes[-1])
                        )

                    # index of the original target box
                    self.target_bbox_index = min(
                        range(len(all_box_sizes)),
                        key=lambda i: abs(all_box_sizes[i] - target_bbox_size),
                    )

            # Load the original image
            image_path = sample["img_path"]
            # self.original_image = Image.open(image_path)
            self.grounded_check = True
            # original_image = self.original_image.convert("RGB")
            original_image = self.image_to_ground.convert("RGB")

            im_width, im_height = original_image.size

            final_images = []  # List to store each modified image

            for bbox in expanded_bboxes:
                # original_image = self.original_image.convert("RGB")

                # original_image = self.draw_bbox(original_image, bbox, color="red", width=3)
                # Apply blur to the entire image
                blurred_image = original_image.filter(
                    ImageFilter.GaussianBlur(radius=15)
                )

                # Create a mask for the expanded bounding box
                left, top, right, bottom = [
                    int(coord)
                    for coord in (
                        bbox[0] * im_width,
                        bbox[1] * im_height,
                        bbox[2] * im_width,
                        bbox[3] * im_height,
                    )
                ]
                mask = np.zeros((im_height, im_width), dtype=np.uint8)
                mask[top:bottom, left:right] = 255
                mask = Image.fromarray(mask)

                # Overlay the clear area over the blurred image
                final_image = Image.composite(original_image, blurred_image, mask)
                final_images.append(final_image)

            # # Save the distorted images
            # saved_paths = []
            # for i, cropped_img in enumerate(final_images, start=1):
            #     save_path = f"decoder_zoo/HaLC/mnt/cropped_level_{i}.png"
            #     cropped_img.save(save_path)
            #     saved_paths.append(save_path)
            # input("img saved!")

            # get decoding for each context window
            max_new_tokens = self.max_new_tokens
            max_length = 2000
            embeds_list = []
            for i, cropped_img in enumerate(final_images, start=1):
                image = self.vis_processor(cropped_img).unsqueeze(0).to(self.device)
                image_emb, _ = self.model.encode_img(image, 38)
                prompt = self.prompt
                # print("prompt: ", prompt)
                embs = self.model.get_context_emb(prompt, [image_emb])
                current_max_len = embs.shape[1] + max_new_tokens

                if current_max_len - max_length > 0:
                    print(
                        "Warning: The number of tokens in current conversation exceeds the max length. "
                        "The model will not see the contexts outside the range."
                    )

                begin_idx = max(0, current_max_len - max_length)
                embs = embs[:, begin_idx:]
                embeds_list.append(embs)

        else:
            detect_info["status"] = "invalid"
            embeds_list = None

        return embeds_list, detect_info

    def naive_focus_decoding(self, context_logits_list):
        """
        directly apply the detected box for decoding
        """
        contrast_logits = context_logits_list[0]
        return False, contrast_logits

    def auto_regressive_decoding(self, context_logits_list):
        """
        directly apply the detected box for decoding
        """
        return True, None

    def context_curve_contrastive_decoding(self, context_logits_list):
        """
        this decoding method use the hallucination pattern for decoding
        """
        target_layer = context_logits_list[0]
        lower_layer = context_logits_list[1]
        upper_layer = context_logits_list[2]

        relative_top = 0.1

        # if relative_top > 0.0:
        #     final_logits = self.relative_top_filter(target_layer, relative_top)
        #     base_logits = upper_layer.log_softmax(dim=-1)
        #     mask = final_logits[0] < -1e3
        #     base_logits[0][mask] = -1e3
        # upper_contrast_logits = final_logits - base_logits

        # if relative_top > 0.0:
        #     final_logits = self.relative_top_filter(target_layer, relative_top)
        #     base_logits = lower_layer.log_softmax(dim=-1)
        #     mask = final_logits[0] < -1e3
        #     base_logits[0][mask] = -1e3
        # lower_contrast_logits = final_logits - base_logits

        upper_contrast_logits = target_layer - upper_layer
        lower_contrast_logits = target_layer - lower_layer

        # find out those tokens that are positive in both upper and lower contrast logits
        # Step 1: Identify positive logits
        positive_upper = upper_contrast_logits > 0
        positive_lower = lower_contrast_logits > 0

        # Step 2: Create a combined mask
        positive_both = np.logical_and(
            positive_upper.cpu().numpy(), positive_lower.cpu().numpy()
        )

        contrast_logits = upper_layer.cpu().numpy() * positive_both

        contrast_logits = torch.tensor(contrast_logits).to(self.device)

        return False, contrast_logits

    def context_contrastive_decoding(self, context_logits_list, last_tokens):
        #
        # this decoding method use the hallucination pattern as a filter for verification
        #
        hallucination_index = last_tokens[0]

        # print("ontext_logits_list[0]", context_logits_list[0])
        target_layer = context_logits_list[0]
        lower_layer = context_logits_list[1]
        upper_layer = context_logits_list[2]

        target_logits = target_layer[0][hallucination_index]
        upper_logits = upper_layer[0][hallucination_index]
        lower_logits = lower_layer[0][hallucination_index]
        upper_contrast_logits = target_logits - upper_logits
        lower_contrast_logits = target_logits - lower_logits

        if upper_contrast_logits > -5 and lower_contrast_logits > -5:
            skip_flag = True
        else:
            skip_flag = False

        # skip_flag = False

        return skip_flag, target_layer

    def relative_top_filter(
        self,
        scores: torch.FloatTensor,
        relative_top: float = 0.1,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.FloatTensor:
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        scores_normalized[scores_normalized < probs_thresh] = filter_value

        return scores_normalized

    def auto_contrastive_context_decoding(self, context_logits_list, last_tokens):
        """
        The method uses a list of context windows rooted from the DINO detection one and apply the contrastive decoding method to each context-window pair to get a list of contrastive logits. Then we use the contrastive logits to do the decoding.
        """
        hallucination_index = last_tokens[0]

        target_layer = context_logits_list[self.target_bbox_index]
        lower_layer = context_logits_list[0]
        upper_layer = context_logits_list[-1]

        target_logits = target_layer[0][hallucination_index]
        upper_logits = upper_layer[0][hallucination_index]
        lower_logits = lower_layer[0][hallucination_index]
        upper_contrast_logits = target_logits - upper_logits
        lower_contrast_logits = target_logits - lower_logits

        # if upper_contrast_logits > -2 and lower_contrast_logits > -2:
        if False:
            skip_flag = True
            return skip_flag, target_layer
        else:
            skip_flag = False

            non_target_layer_indices = [
                i
                for i in range(len(context_logits_list))
                if i != self.target_bbox_index
            ]
            # 1. Stacking all non-target context layer into a new dimension
            stacked_premature_layers = torch.stack(
                [context_logits_list[i] for i in non_target_layer_indices],
                dim=0,
            )

            # 2. Calculate the softmax values for mature_layer and all premature_layers
            softmax_mature_layer = F.softmax(
                context_logits_list[self.target_bbox_index], dim=-1
            )  # shape: (batch_size, num_features)
            softmax_premature_layers = F.softmax(
                stacked_premature_layers, dim=-1
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 3. Calculate M, the average distribution
            M = 0.5 * (
                softmax_mature_layer[None, :, :] + softmax_premature_layers
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 4. Calculate log-softmax for the KL divergence
            log_softmax_mature_layer = F.log_softmax(
                context_logits_list[self.target_bbox_index], dim=-1
            )  # shape: (batch_size, num_features)
            log_softmax_premature_layers = F.log_softmax(
                stacked_premature_layers, dim=-1
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 5. Calculate the KL divergences and then the JS divergences
            kl1 = F.kl_div(
                log_softmax_mature_layer[None, :, :], M, reduction="none"
            ).mean(
                -1
            )  # shape: (num_premature_layers, batch_size)
            kl2 = F.kl_div(log_softmax_premature_layers, M, reduction="none").mean(
                -1
            )  # shape: (num_premature_layers, batch_size)
            js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

            # 6. Reduce the batchmean
            js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)

            premature_layer_index = non_target_layer_indices[
                int(js_divs.argmax().cpu().item())
            ]

            base_logits = context_logits_list[premature_layer_index]
            final_logits = context_logits_list[self.target_bbox_index]

            # final_logits = self.relative_top_filter(final_logits, relative_top=0.1)
            # base_logits = base_logits.log_softmax(dim=-1)
            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3
            contrast_logits = final_logits - base_logits

            # we always use the target layer for decoding
            return skip_flag, contrast_logits

    def contrastive_avg_context_decoding(self, context_logits_list, last_tokens):
        hallucination_index = last_tokens[0]

        target_layer = context_logits_list[self.target_bbox_index]
        lower_layer = context_logits_list[0]
        upper_layer = context_logits_list[-1]

        target_logits = target_layer[0][hallucination_index]
        upper_logits = upper_layer[0][hallucination_index]
        lower_logits = lower_layer[0][hallucination_index]
        upper_contrast_logits = target_logits - upper_logits
        lower_contrast_logits = target_logits - lower_logits

        if upper_contrast_logits > -2 and lower_contrast_logits > -2:
            skip_flag = True
            return skip_flag, target_layer
        else:
            skip_flag = False

            non_target_layer_indices = [
                i
                for i in range(len(context_logits_list))
                if i != self.target_bbox_index
            ]
            # stack all non-target context layer into a new dimension
            stacked_premature_layers = torch.stack(
                [context_logits_list[i] for i in non_target_layer_indices],
                dim=0,
            )

            # compute the average of premature layers
            premature_layer_avg = stacked_premature_layers.mean(dim=0)
            final_logits = context_logits_list[self.target_bbox_index]
            contrast_logits = final_logits - premature_layer_avg

            # we always use the target layer for decoding
            return skip_flag, contrast_logits

    def context_layer_contrastive_decoding(self, context_logits_list, last_tokens):
        """
        The method uses a list of context windows rooted from the DINO detection one and apply the contrastive decoding method to each context-window pair to get a list of contrastive logits. Then we use the contrastive logits to do the decoding.
        """
        skip_flag = False

        all_layer_indices = range(len(context_logits_list))

        # 1. Stacking all non-target context layer into a new dimension
        stacked_premature_layers = torch.stack(
            [context_logits_list[i] for i in all_layer_indices],
            dim=0,
        )

        # print("stacked_premature_layers", np.shape(stacked_premature_layers))
        # input()
        num_layers = len(stacked_premature_layers)
        jsd_matrix = torch.zeros((num_layers, num_layers))

        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                M = 0.5 * (
                    F.softmax(stacked_premature_layers[i], dim=-1)
                    + F.softmax(stacked_premature_layers[j], dim=-1)
                )
                kl1 = F.kl_div(
                    F.log_softmax(stacked_premature_layers[i], dim=-1),
                    M,
                    reduction="batchmean",
                )
                kl2 = F.kl_div(
                    F.log_softmax(stacked_premature_layers[j], dim=-1),
                    M,
                    reduction="batchmean",
                )
                jsd = 0.5 * (kl1 + kl2)
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd  # Symmetric matrix

        # Find indices of max JSD
        max_jsd_flat_index = torch.argmax(jsd_matrix.triu(diagonal=1))  # .unbind()
        # layer_idx1, layer_idx2 = max_jsd_indices[0], max_jsd_indices[1]
        layer_idx1, layer_idx2 = np.unravel_index(
            max_jsd_flat_index.cpu().numpy(), jsd_matrix.shape
        )
        print("base_layer, final_layer: ", layer_idx1, layer_idx2)

        # # Update final_logits and base_logits
        # final_logits = context_logits_list[layer_idx1]
        # base_logits = context_logits_list[layer_idx2]

        context_domain = self.halc_params["context_domain"]
        # Update final_logits and base_logits
        if context_domain == "upper":
            base_logits = context_logits_list[layer_idx1]
            final_logits = context_logits_list[layer_idx2]
        elif context_domain == "lower":
            base_logits = context_logits_list[layer_idx2]
            final_logits = context_logits_list[layer_idx1]
        else:
            raise ValueError("Invalid context domain!")

        # final_logits = self.relative_top_filter(final_logits, relative_top=0.1)
        # base_logits = base_logits.log_softmax(dim=-1)
        mask = final_logits[0] < -1e3
        base_logits[0][mask] = -1e3

        contrast_weight = self.halc_params["contrast_weight"]

        # contrast_logits = final_logits - base_logits * 0.05
        contrast_logits = final_logits - base_logits * contrast_weight

        # we always use the target layer for decoding

        return skip_flag, contrast_logits

    def context_layer_multi_contrastive_decoding(
        self, context_logits_list, last_tokens
    ):
        """
        The method uses a list of context windows rooted from the DINO detection one and apply the contrastive decoding method to each context-window pair to get a list of contrastive logits. Then we use the contrastive logits to do the decoding.
        """
        skip_flag = False
        k_candidate = self.k_candidate_num
        all_layer_indices = range(len(context_logits_list))

        stacked_premature_layers = torch.stack(
            [context_logits_list[i] for i in all_layer_indices],
            dim=0,
        )

        num_layers = len(stacked_premature_layers)
        jsd_matrix = torch.zeros((num_layers, num_layers))

        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                M = 0.5 * (
                    F.softmax(stacked_premature_layers[i], dim=-1)
                    + F.softmax(stacked_premature_layers[j], dim=-1)
                )
                kl1 = F.kl_div(
                    F.log_softmax(stacked_premature_layers[i], dim=-1),
                    M,
                    reduction="batchmean",
                )
                kl2 = F.kl_div(
                    F.log_softmax(stacked_premature_layers[j], dim=-1),
                    M,
                    reduction="batchmean",
                )
                jsd = 0.5 * (kl1 + kl2)
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd  # Symmetric matrix

        # Find indices of top k_candidate JSD values
        upper_tri_flat = jsd_matrix.triu(diagonal=1).flatten()
        top_k_indices_flat = torch.topk(upper_tri_flat, k_candidate).indices
        rows = top_k_indices_flat // jsd_matrix.size(1)
        cols = top_k_indices_flat % jsd_matrix.size(1)
        top_k_indices = list(zip(rows.tolist(), cols.tolist()))

        contrast_logits_array = []
        context_domain = self.halc_params["context_domain"]
        contrast_weight = self.halc_params["contrast_weight"]

        for layer_idx1, layer_idx2 in top_k_indices:
            # print("base_layer, final_layer: ", layer_idx1, layer_idx2)
            # (layer_idx1, layer_idx2) = top_k_indices[0]

            # Update final_logits and base_logits
            if context_domain == "upper":
                base_logits = context_logits_list[layer_idx1]
                final_logits = context_logits_list[layer_idx2]
            elif context_domain == "lower":
                base_logits = context_logits_list[layer_idx2]
                final_logits = context_logits_list[layer_idx1]
            else:
                raise ValueError("Invalid context domain!")

            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3

            # contrast_logits = final_logits - base_logits * 0.05
            contrast_logits = final_logits - base_logits * contrast_weight
            contrast_logits_array.append(contrast_logits)

        return skip_flag, contrast_logits_array

    def context_layer_double_multi_contrastive_decoding(
        self, context_logits_list, last_tokens
    ):
        """
        The method uses a list of context windows rooted from the DINO detection one and apply the contrastive decoding method to each context-window pair to get a list of contrastive logits. Then we use the contrastive logits to do the decoding.
        """
        skip_flag = False
        if self.k_candidate_num % 2 != 0:
            raise ValueError("k_candidate_num must be even!")
        k_candidate = int(self.k_candidate_num / 2)
        all_layer_indices = range(len(context_logits_list))

        stacked_premature_layers = torch.stack(
            [context_logits_list[i] for i in all_layer_indices],
            dim=0,
        )

        num_layers = len(stacked_premature_layers)
        jsd_matrix = torch.zeros((num_layers, num_layers))

        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                M = 0.5 * (
                    F.softmax(stacked_premature_layers[i], dim=-1)
                    + F.softmax(stacked_premature_layers[j], dim=-1)
                )
                kl1 = F.kl_div(
                    F.log_softmax(stacked_premature_layers[i], dim=-1),
                    M,
                    reduction="batchmean",
                )
                kl2 = F.kl_div(
                    F.log_softmax(stacked_premature_layers[j], dim=-1),
                    M,
                    reduction="batchmean",
                )
                jsd = 0.5 * (kl1 + kl2)
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd  # Symmetric matrix

        # Find indices of top k_candidate JSD values
        upper_tri_flat = jsd_matrix.triu(diagonal=1).flatten()
        top_k_indices_flat = torch.topk(upper_tri_flat, k_candidate).indices
        rows = top_k_indices_flat // jsd_matrix.size(1)
        cols = top_k_indices_flat % jsd_matrix.size(1)
        top_k_indices = list(zip(rows.tolist(), cols.tolist()))

        contrast_logits_array = []
        context_domain = self.halc_params["context_domain"]
        contrast_weight = self.halc_params["contrast_weight"]

        for layer_idx1, layer_idx2 in top_k_indices:
            base_logits = context_logits_list[layer_idx1]
            final_logits = context_logits_list[layer_idx2]

            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3

            contrast_logits = final_logits - base_logits * contrast_weight
            contrast_logits_array.append(contrast_logits)

            base_logits = context_logits_list[layer_idx2]
            final_logits = context_logits_list[layer_idx1]

            mask = final_logits[0] < -1e3
            base_logits[0][mask] = -1e3

            contrast_logits = final_logits - base_logits * contrast_weight
            contrast_logits_array.append(contrast_logits)

        return skip_flag, contrast_logits_array

    def clip_score_selection(
        self, candidate_intermediate_token_lists_array, beam_size, skip_token_length=0
    ):
        if (
            candidate_intermediate_token_lists_array
            == [None] * len(candidate_intermediate_token_lists_array)
            or self.grounded_check == False
        ):
            # print("identical candidate lists: ", candidate_intermediate_token_lists_array)
            random.seed(8)
            selected_candidates = random.sample(
                range(len(candidate_intermediate_token_lists_array)), beam_size
            )

        else:
            candidate_texts = []
            for (
                candidate_intermediate_token_lists
            ) in candidate_intermediate_token_lists_array:
                # print("candidate_intermediate_token_lists[0]", candidate_intermediate_token_lists[0])
                if (
                    self.model_backbone == "minigpt4"
                    or self.model_backbone == "instructblip"
                ):
                    skip_token_length = 0
                elif (
                    self.model_backbone == "llava-1.5"
                    or self.model_backbone == "mplug-owl2"
                ):
                    skip_token_length = skip_token_length

                    # print("tokens_to_text", tokens_to_text)
                candidate_texts.append(
                    self.get_sequence_text(
                        candidate_intermediate_token_lists[0], skip_token_length
                    )
                )

            # print("candidate_texts", candidate_texts)
            # input()
            # original_image = self.original_image
            original_image = self.image_to_ground

            if self.score_type == "CLIP" or self.score_type == "BLIP":
                # print("candidate_texts", candidate_texts)
                clip_inputs = self.score_processor(
                    text=candidate_texts,
                    images=original_image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                clip_outputs = self.score_model(**clip_inputs)
                logits_per_image = (
                    clip_outputs.logits_per_image
                )  # image-text similarity score
                clip_probs = logits_per_image.softmax(dim=1)[
                    0
                ]  # take the softmax to get the label probabilities

                # print("candidate lists:", candidate_intermediate_token_lists_array)
                # print("clip_probs:", clip_probs)

                # # get the top beam_size candidates
                clip_probs = clip_probs.cpu().numpy()
                # candidate_index = clip_probs.argsort()[-beam_size:][::-1]
                scores = clip_probs

            elif self.score_type == "HPSv2":
                imgs_path = self.original_image
                scores = hpsv2.score(imgs_path, candidate_texts, hps_version="v2.1")
                scores = np.array(scores).tolist()

            elif self.score_type == "Random":
                scores = np.random.rand(len(candidate_texts)).tolist()

            elif self.score_type == "Perplexity":
                


            sorted_indices = scores.argsort()[
                -len(candidate_intermediate_token_lists_array) :
            ][::-1]
            selected_texts = set()
            selected_candidates = []
            # print("sorted_indices:", sorted_indices)
            for idx in sorted_indices:
                # candidate_text = self.get_sequence_text(candidate_intermediate_token_lists_array[idx][0])
                candidate_text = candidate_texts[idx]
                # Check for uniqueness
                if candidate_text not in selected_texts:
                    selected_texts.add(candidate_text)
                    selected_candidates.append(idx)

                # Stop if enough candidates have been selected
                if len(selected_candidates) == beam_size:
                    break

        # print("selected_candidates:", selected_candidates)
        if len(selected_candidates) < beam_size:
            # copy the first one
            for _ in range(beam_size - len(selected_candidates)):
                selected_candidates.append(selected_candidates[0])

        candidate_index = selected_candidates
        # print("candidate_index:", candidate_index)
        # input()

        return candidate_index

    def random_selection(self, candidate_intermediate_token_lists_array, beam_size):
        random.seed(8)
        candidate_index = random.sample(
            range(len(candidate_intermediate_token_lists_array)), beam_size
        )

        return candidate_index
