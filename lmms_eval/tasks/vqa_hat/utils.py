"""VQA-HAT task utilities.

Reuses VQAv2's exact_match evaluation logic (EvalAIAnswerProcessor).
"""

import statistics

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"Expected result list of length 1, got {len(result)}."
    resAns = result[0]
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        for ansDic in doc["answers"]:
            ansDic["answer"] = ansDic["answer"].replace("\n", " ")
            ansDic["answer"] = ansDic["answer"].replace("\t", " ")
            ansDic["answer"] = ansDic["answer"].strip()
        resAns = resAns.replace("\n", " ")
        resAns = resAns.replace("\t", " ")
        resAns = resAns.strip()

        for ansDic in doc["answers"]:
            ansDic["answer"] = eval_ai_processor.process_punctuation(ansDic["answer"])
            ansDic["answer"] = eval_ai_processor.process_digit_article(ansDic["answer"])
        resAns = eval_ai_processor.process_punctuation(resAns)
        resAns = eval_ai_processor.process_digit_article(resAns)

        gtAcc = []
        for gtAnsDatum in doc["answers"]:
            otherGTAns = [item for item in doc["answers"] if item != gtAnsDatum]
            matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
    }
