import torch
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from sklearn.metrics import confusion_matrix

def compute_metrics(model, data_module):
    normalized_levenshtein = NormalizedLevenshtein()
    nls_by_item = []

    tp, fp, tn, fn = 0, 0, 0, 0
    for batch in data_module.predict_dataloader():
        item = batch[0]

        predict_ids_list = get_predict_ids_list(model, batch)

        original_text = get_text(item["input_ids"])
        predicted_text = get_text(predict_ids_list)

        nls = normalized_levenshtein.similarity(original_text, predicted_text)
        nls_by_item.append(nls)

        tp_, fp_, tn_, fn_ = simple_multiclass_conf_matrix(item["input_ids"], predict_ids_list)
        tp += tp_
        fp += fp_
        tn += tn_
        fn += fn_

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    metrics = {
        "anls": nls_by_item/len(nls_by_item),
        "precision": precision,
        "recall": recall,
        "f-score": 2 * precision * recall / (precision + recall)
    }

    return metrics


def get_text(token_ids, tokenizer):
    return tokenizer.convert_tokens_to_string(token_ids)


def get_predict_ids_list(model, encodings):
    with torch.no_grad():
        output = model(**encodings)
    predict_ids = torch.topk(output.logits[0], 1, dim=1).indices.tolist()

    predict_ids_list = []
    for ids in predict_ids:
        predict_ids_list.append(ids[0])

    return predict_ids_list

def simple_multiclass_conf_matrix(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return (TP, FP, TN, FN)

