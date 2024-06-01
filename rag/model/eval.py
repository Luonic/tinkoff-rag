import torch
from datasets import load_metric
import torch.nn.functional as F
from sklearn.metrics import f1_score as sklearn_f1_score
from time import perf_counter

def average_metrics(metrics):
    # average BERTScore
    bertscore = metrics["BERTScore"]
    average_bertscore = ['precision', 'recall', 'f1']
    bertscore_avg = {}
    for metric in average_bertscore:
        bertscore_avg[metric] = sum(bertscore[metric]) / len(bertscore[metric])

    # average ROUGE 
    ROUGE = metrics["ROUGE"]

    # ROUGE['rouge1'][1][2] = ROUGE['rouge1']['mid']['fmeasure']
    rouge_avg = {
        'rouge1': ROUGE['rouge1'][1][2],
        'rouge2': ROUGE['rouge2'][1][2],
        'rougeL': ROUGE['rougeL'][1][2],
        'rougeLsum': ROUGE['rougeLsum'][1][2],
    }

    # average BLEU
    BLEU = metrics["BLEU"]
    bleu_avg = {'score': BLEU["score"]}

    # CEloss and F1
    all_metrics = {
        'f1': metrics["F1"],
        'CEloss': metrics["Cross-Entropy Loss"],
    }

    write_metrics = {"BLEU": bleu_avg, "ROUGE": rouge_avg, "BERTScore": bertscore_avg}
    for metric_name, values in write_metrics.items():
        for submetric_name, submetric_value in values.items():
            all_metrics[f"{metric_name}/{submetric_name}"] = submetric_value
    
    return all_metrics

def evaluate(model, test_dataloader, tokenizer):
    metrics_start = perf_counter()
    model.eval()
    
    # Load metrics
    bleu_metric = load_metric("sacrebleu")
    rouge_metric = load_metric("rouge")
    bertscore_metric = load_metric("bertscore", trust_remote_code=True)
    
    total_loss = 0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    for batch in test_dataloader:
        inputs = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['input_ids'].to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        
        total_loss += loss.item()
        num_batches += 1
        
        # Decode predictions and references
        predictions = torch.argmax(logits, dim=-1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Extract the part of the text that begins with [INST/]
        decoded_preds = [pred.split('[INST/]')[-1] for pred in decoded_preds]
        decoded_labels = [label.split('[INST/]')[-1] for label in decoded_labels]
        
        # Tokenize the decoded predictions and labels
        tokenized_preds = [tokenizer.tokenize(pred) for pred in decoded_preds]
        tokenized_labels = [tokenizer.tokenize(label) for label in decoded_labels]
        
        # Ensure predictions and labels have the same length by padding or truncating
        max_len = max(max(len(pred) for pred in tokenized_preds), max(len(label) for label in tokenized_labels))
        tokenized_preds = [pred + [''] * (max_len - len(pred)) for pred in tokenized_preds]
        tokenized_labels = [label + [''] * (max_len - len(label)) for label in tokenized_labels]
        
        # Flatten the tokenized predictions and labels
        flat_preds = [token for sublist in tokenized_preds for token in sublist]
        flat_labels = [token for sublist in tokenized_labels for token in sublist]
        
        all_preds.extend(flat_preds)
        all_labels.extend(flat_labels)
        
        # Compute metrics
        bleu_metric.add_batch(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        bertscore_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        torch.cuda.empty_cache()
    
    # Compute final metrics
    bleu_score = bleu_metric.compute()
    rouge_score = rouge_metric.compute()
    bertscore = bertscore_metric.compute(lang="en") # model_type="roberta-large"

    avg_loss = total_loss / num_batches
    
    # Compute token-level F1 score using sklearn
    f1_score = sklearn_f1_score(all_labels, all_preds, average='weighted')
    
    results = {
        "BLEU": bleu_score,
        "ROUGE": rouge_score,
        "F1": f1_score,
        "Cross-Entropy Loss": avg_loss,
        "BERTScore": bertscore
    }
    averaged = average_metrics(results)
    metrics_end = perf_counter()
    averaged["test_compute_time"] = metrics_end - metrics_start
    model.train()

    return averaged
