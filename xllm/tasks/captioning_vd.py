"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from xllm.common.dist_utils import main_process
from xllm.common.registry import registry
from xllm.tasks.base_task import BaseTask
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import jieba

@registry.register_task("captioning_vd")
class CaptionVDTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )
        # questions = samples["image_id"]
        # captions = [_.replace("\\n", "") for _ in captions]
        gt_anwers = samples["text_output"]
        for caption, gt in zip(captions, gt_anwers):
            results.append({"gt_answer": gt, "pred_answer": caption})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="",
        )

        # if self.report_metric:
        #     metrics = self._report_metrics(
        #         eval_result_file=val_result, split_name=split_name
        #     )
        # else:
        metrics = {"agg_metrics": 0.0}

        return metrics
    


    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        def cal_bleu(pred, target):
    
            pred = [list(p.lower()) for p in pred]
            target = [[list(t.lower())] for t in target]

            def cal(targets, preds, n_gram):
                weights = [1/n_gram] * n_gram + [0] * (4-n_gram)
                score = corpus_bleu(targets, preds, weights=weights)
                return score
            scores = [cal(target, pred, x)  for x in range(1,5)]
            return scores
        
        hyps = []
        refs = []
        for item in eval_result_file:
            hyps.append(item['pred_answer'])
            refs.append(item['gt_answer'])

        bleu_scores = cal_bleu(hyps, refs)
        logging_output = {}
        logging_output['bleu1'] = bleu_scores[0]
        logging_output['bleu2'] = bleu_scores[1]
        logging_output['bleu3'] = bleu_scores[2]
        logging_output['bleu4'] = bleu_scores[3]

        agg_metrics = bleu_scores[3] + bleu_scores[1] + bleu_scores[2] + bleu_scores[0]
        log_stats = {split_name: {k: v for k, v in logging_output.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in logging_output.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
