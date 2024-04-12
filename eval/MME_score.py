import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import json

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results_dir', default=None, type=str)
parser.add_argument('-g', '--gt_dir', default=None, type=str)
parser.add_argument('-s', '--save', action='store_true', default=False)


eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            True: 1,
            False: 0,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, results_dict, gt_dict, task_name):

        caption_score = {}
        img_num = len(gt_dict)

        # print("results_dict", results_dict.keys())
        # input()
        results_list = []
        for caption_file in results_dict:
            results_list.append(results_dict[caption_file])


            caption_score[caption_file] = {}
            caption_dict = results_dict[caption_file]
            gts = []
            preds = []
            task_score = 0
            acc_plus_correct_num = 0
            img_plus_correct_num = 0
            scores = 0
            for img_id, gt_items in gt_dict.items():
                # print("img_id", img_id)
                # print("gt_items", gt_items)

                assert len(gt_items) == 2
                img_correct_num = 0

                for img_item in gt_items:
                    img_name = img_id
                    # if img_name not in results_dict:
                    #     continue
                    # print("caption_dict", caption_dict)
                    pred_caption = caption_dict[img_name]
                    gt_question = img_item
                    gt_ans = gt_items[gt_question]

                    pred_caption = pred_caption.lower()
                    if gt_question.lower() in pred_caption:
                        pred_ans = True
                    else:
                        pred_ans = False
                    # print("pred_caption", pred_caption)
                    # print("pred_ans", pred_ans)
                    # print("gt_question", gt_question)
                    # print("gt_ans", gt_ans)
                    # print("pred_ans", pred_ans)
                    # input()

                    assert gt_ans in [True, False] # gt can only be yes or no.

                    assert pred_ans in [True, False]

                    gts.append(gt_ans)
                    preds.append(pred_ans)
                    
                    if gt_ans == pred_ans:
                        if gt_ans == True:
                            img_correct_num += 1
                            img_plus_correct_num += 1
            
                        elif gt_ans == False:
                            img_correct_num += 1
                            img_plus_correct_num += 1
                    if pred_ans not in [True,False]:
                        print("pred_ans", pred_ans)
                        input("NOT EXIST")
                        # task_other_ans_num += 1

                if img_correct_num == 2:
                    acc_plus_correct_num += 1
                    acc_plus_correct_flag = 1
                else:
                    acc_plus_correct_flag = 0

                caption_score[caption_file][img_name] = {"img_correct_num": img_correct_num}
                caption_score[caption_file][img_name]["acc_plus_correct_flag"] = acc_plus_correct_flag
                caption_score[caption_file][img_name]["acc_plus_correct_num"] = acc_plus_correct_num
            # cal TP precision acc, etc.
            metric_dict = self.compute_metric(gts, preds)
            # print("acc", metric_dict["acc"])
            metric_dict = {}
            metric_dict["acc"] = img_plus_correct_num / (img_num * 2)
            # print("acc", metric_dict["acc"])
            acc_plus = acc_plus_correct_num / img_num
            metric_dict["acc_plus"] = acc_plus
            
            for k, v in metric_dict.items():
                if k in ["acc", "acc_plus"]:
                    task_score += v*100
            scores += task_score
            # print(f"{caption_file} score:", task_score, "\n")
            # print(f"{task_name} score:", scores, "\n")
            # print("caption_score", caption_score)
            # input()

        
        return caption_score, results_list




if __name__ == "__main__":
    cal = calculate_metrics()

    args = parser.parse_args()
    results_dir = args.results_dir
    save = args.save

    # subset_dir = ["MME_benchmark/MME_dataset/existence_modified", "MME_benchmark/MME_dataset/position_modified",
    #                "MME_benchmark/MME_dataset/color_modified", "MME_benchmark/MME_dataset/count_modified"]
    
    subset_dir = ["MME_benchmark/MME_dataset/position_modified"]


    results_dict = {}
    if "generated_captions" not in results_dir:
        caption_filenames = os.listdir(results_dir)
        for caption_file in caption_filenames:
            if "halc" not in caption_file:
                continue
            caption_dict = {}
            with open(results_dir + "/" + caption_file, "r") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    line = json.loads(line)
                    img_id = line["image_id"]
                    caption = line["caption"]
                    caption_dict[img_id] = caption
            results_dict[caption_file] = caption_dict
    else:
        with open(results_dir, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = json.loads(line)
                img_id = line["image_id"]
                caption = line["caption"]
                results_dict[img_id] = caption

        results_dict = {results_dir: results_dict}
    
    # print("results_dict", results_dict.keys())
    # input()
    merge_list = []
    merge_key = []
    for gt_dir in subset_dir:
        txt_filenames = os.listdir(gt_dir)
        # Initialize an empty dictionary to hold the data
        gt_dict = {}

        # Process each .txt file
        for txt_file in txt_filenames:
            if not txt_file.endswith(".txt"):
                continue
            # Extract the image ID from the .txt filename
            img_id = int(txt_file.split(".txt")[0][-6:])
            
            # Initialize a dictionary for the current image ID
            gt_dict[img_id] = {}

            # Construct the full path to the .txt file and read it
            with open(os.path.join(gt_dir, txt_file), 'r') as file:
                lines = file.readlines()
                # Assuming there are two lines, one for positive and one for negative word
                for line in lines:
                    word, value = line.strip().split(': ')
                    # Convert the string to a boolean value
                    gt_dict[img_id][word] = True if value.lower() == 'yes' else False

        # The word_dict now contains the desired structure
    
        task_name = gt_dir.split("/")[-1].split("_")[0]
        # print(task_name, len(gt_dict))
        caption_score_list = []
        print("gt_dict", gt_dict)
        caption_score, results_list = cal.process_result(results_dict, gt_dict, task_name)

        # for caption_name, caption_score in caption_score.items():
        #     task_score = 0
        #     for img_id, img_score in caption_score.items():
        #         img_num = len(caption_score)
        #         img_correct_score = img_score["img_correct_num"] / (img_num * 2)
        #         acc_plus_correct_flag = img_score["acc_plus_correct_flag"]
        #         acc_plus_score = acc_plus_correct_flag / img_num
                
        #         task_score += img_correct_score * 100
        #         task_score += acc_plus_score * 100
        
        # print("task_score", task_score)
        # input()

        for caption_name, caption_score in caption_score.items():
            task_score = 0
            for img_id, img_score in caption_score.items():
                img_num = len(caption_score)
                img_correct_score = img_score["img_correct_num"] / (img_num * 2)
                acc_plus_correct_flag = img_score["acc_plus_correct_flag"]
                acc_plus_score = acc_plus_correct_flag / img_num
                
                # task_score += img_correct_score * 100
                # task_score += acc_plus_score * 100

                caption_score[img_id] = img_correct_score * 100 + acc_plus_score * 100
                task_score += caption_score[img_id]

            caption_score_list.append(caption_score)
            print("caption_score", caption_score)
            print("task_score", task_score)
