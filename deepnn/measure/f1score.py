import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class F1Score():
    def __init__(self):
        pass
    def __call__(self, output, ground_truth, threshold, eps=1e-8):
        with torch.no_grad():
            predictions = (output.cpu().numpy() > threshold).astype(int)
            ground_truth = (ground_truth.cpu().numpy() > threshold).astype(int)

            # Scikit f1 score implementation
            prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(ground_truth, predictions, average='samples')
            acc_a = accuracy_score(ground_truth, predictions)
            return f1_a, prec_a, rec_a, acc_a
            #return precision.mean(), recall.mean(), f1scr.mean()

            # Self implementation of precision, recall, f1 and accuracy
            #actual_pos = ground_truth.sum(axis=1)
            #predic_pos = predictions.sum(axis=1)
            #true_pos = ((ground_truth == predictions) * (ground_truth == 1)).sum(axis=1)
            #prec = true_pos / (actual_pos+eps)
            #rec = true_pos / (predic_pos+eps)
            
            #denom = (prec + rec)
            #denom[denom==0] = eps
            #f1 = 2 * (prec * rec) / denom
            
            #ncorrect_pred = (predictions == ground_truth).sum()
            #ntotal = ground_truth.size
            #acc =  ncorrect_pred / ntotal
         
            #return f1.mean(), prec.mean(), rec.mean(), acc.mean()

