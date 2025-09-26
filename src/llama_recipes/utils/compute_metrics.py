from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from netcal.presentation import ReliabilityDiagram
import numpy as np
from netcal.metrics import ECE
import wandb
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skm
from io import BytesIO
from PIL import Image

class SimpleStatsCache:
    def __init__(self, confids, correct):
        self.confids = np.array(confids)
        self.correct = np.array(correct)

    @property
    def rc_curve_stats(self):
        coverages = []
        risks = []

        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)

        coverage = n_residuals
        error_sum = sum(self.residuals[idx_sorted])

        coverages.append(coverage / n_residuals)
        risks.append(error_sum / n_residuals)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.residuals[idx_sorted[i]]
            selective_risk = error_sum / (n_residuals - 1 - i)
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_residuals)

        return coverages, risks, weights

    @property
    def residuals(self):
        return 1 - self.correct

def area_under_risk_coverage_score(confids, correct):
    stats_cache = SimpleStatsCache(confids, correct)
    _, risks, weights = stats_cache.rc_curve_stats
    AURC_DISPLAY_SCALE = 1000
    return sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])* AURC_DISPLAY_SCALE

def plot_confidence_histogram(y_true, y_confs, score_type, acc, auroc, ece, wandb_run, original, dataset, use_annotation=True):

    plt.figure(figsize=(6, 4))    
    corr_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 1]
    wrong_confs = [y_confs[i]*100 for i in range(len(y_confs)) if y_true[i] == 0]

    corr_counts = [corr_confs.count(i) for i in range(101)]
    wrong_counts = [wrong_confs.count(i) for i in range(101)]

    # correct_color = plt.cm.tab10(0)
    # wrong_color = plt.cm.tab10(1)
    correct_color =  plt.cm.tab10(0)
    wrong_color = plt.cm.tab10(3)
    # plt.bar(range(101), corr_counts, alpha=0.5, label='correct', color='blue')
    # plt.bar(range(101), wrong_counts, alpha=0.5, label='wrong', color='orange')
    n_wrong, bins_wrong, patches_wrong = plt.hist(wrong_confs, bins=21, alpha=0.8, label='wrong answer', color=wrong_color, align='mid', range=(-2.5,102.5))
    n_correct, bins_correct, patches_correct = plt.hist(corr_confs, bins=21, alpha=0.8, label='correct answer', color=correct_color, align='mid', range=(-2.5,102.5), bottom=np.histogram(wrong_confs, bins=21, range=(-2.5,102.5))[0])

    tick_set = [i*10 for i in range(5,11)]
    annotation_correct_color = "black"
    annotation_wrong_color = "red"
    annotation_texts = []

    # for i, count in enumerate(corr_counts):
    #     if count == 0:
    #         continue
    #     if use_annotation:
    #         annotation_texts.append(plt.annotate(str(count), xy=(i, count), ha='center', va='bottom', c=annotation_correct_color, fontsize=10, fontweight='bold'))
    #     tick_set.append(i)
            
    # for i, count in enumerate(wrong_counts):
    #     if count == 0:
    #         continue
    #     if use_annotation:
    #         annotation_texts.append(plt.annotate(str(count), xy=(i, count), ha='center', va='bottom', c=annotation_wrong_color, fontsize=10, fontweight='bold'))
    #     tick_set.append(i)
    # adjust_text(annotation_texts, only_move={'text': 'y'})

    plt.title(f"ACC {acc:.2f} / AUROC {auroc:.2f} / ECE {ece:.2f}", fontsize=16, fontweight='bold')
    
    
    # plt.xlim(47.5, 102.5)
    plt.ylim(0, 1.1*max(n_correct+n_wrong))
    plt.xticks(tick_set, fontsize=16, fontweight='bold')
    plt.yticks([])
    plt.xlabel("Confidence (%)", fontsize=16, fontweight='bold')
    plt.ylabel("Count", fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', prop={'weight':'bold', 'size':16})
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    # 记录图像到WandB
    if original:
        wandb_run.log({f"plots/original/{dataset}/auroc": wandb.Image(img)})
    else:
        wandb_run.log({f"plots/fine-tuned/{dataset}/auroc": wandb.Image(img)})
    plt.show()
    # plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}.png")), dpi=600)
    # plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_auroc_{score_type}.pdf")), dpi=600)

# def plot_ece_diagram(y_true, y_confs, score_type, wandb_run, original, dataset):
#     from netcal.presentation import ReliabilityDiagram
#     n_bins = 10
#     diagram = ReliabilityDiagram(n_bins)

#     plt.figure(figsize=(5,20))
#     diagram.plot(np.array(y_confs), np.array(y_true))
#     plt.rcParams.update({'font.size': 12})
#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     img = Image.open(buf)

#     # 记录图像到WandB
#     if original:
#         wandb_run.log({f"plots/original/{dataset}/ece": wandb.Image(img)})
#     else:
#         wandb_run.log({f"plots/fine-tuned/{dataset}/ece": wandb.Image(img)})
#     plt.show()
#     #plt.savefig(osp.join(visual_folder, input_file_name.replace(".json", f"_ece_{score_type}.pdf")), dpi=600)

def plot_ece_diagram(y_true, y_confs, score_type, wandb_run, original, dataset):
    """
    Draw a custom reliability diagram (ECE plot) similar to the reference image.
    This diagram shows the relationship between model confidence and actual accuracy,
    highlighting calibration gaps.
    
    Args:
        y_true: list of ground truth labels (1 for correct, 0 for incorrect)
        y_confs: list of confidence scores (between 0 and 1)
        score_type: type of score being visualized
        wandb_run: wandb run object for logging
        original: whether this is for original model or fine-tuned model
        dataset: dataset name
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_confs = np.array(y_confs)
    
    # Create figure
    plt.figure(figsize=(5, 4))
    
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    
    # Define bins for grouping confidence scores
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)  # Bin boundaries
    bin_width = 1.0 / n_bins
    
    # Initialize arrays to store bin statistics
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Group predictions by confidence and calculate accuracy for each bin
    for i in range(len(y_confs)):
        # Determine which bin this confidence value belongs to
        if y_confs[i] == 1.0:  # Handle the edge case for confidence = 1.0
            bin_idx = n_bins - 1
        else:
            bin_idx = int(y_confs[i] / bin_width)
            
        bin_counts[bin_idx] += 1
        bin_accuracies[bin_idx] += y_true[i]
    
    # Calculate average accuracy for each bin
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_accuracies[i] /= bin_counts[i]
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='red',linewidth=2)
    
    # Plot bars for each bin
    for i in range(n_bins):
        # Skip bins with no data
        if bin_counts[i] == 0:
            continue
            
        # The bin center for plotting (x-axis position)
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        
        # The expected confidence for this bin is the bin center
        expected_confidence = bin_center
        
        # The actual accuracy in this bin
        accuracy = bin_accuracies[i]
        
        # Draw the output (blue) part - representing the measured accuracy
        plt.bar(bin_center, min(accuracy, expected_confidence), width=bin_width, color='#3274A1', 
                align='center')
        
        # If expected confidence > accuracy, draw the gap (red) part - overconfidence
        if expected_confidence > accuracy:
            # The gap is the difference between expected confidence and actual accuracy
            gap = expected_confidence - accuracy
            plt.bar(bin_center, gap, width=bin_width, bottom=accuracy, 
                    color='#E15759', align='center', )
        # If expected confidence < accuracy, draw the gap (green) part - underconfidence
        elif expected_confidence < accuracy:
            # The gap is the difference between actual accuracy and expected confidence
            gap = accuracy - expected_confidence
            plt.bar(bin_center, gap, width=bin_width, bottom=expected_confidence, 
                    color='#CB0404', align='center')
    
    # Add title and labels
    # TODO
    # plt.title('TriviaQA', fontsize=18, fontweight='bold')
    plt.xlabel('Confidence', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=18, fontweight='bold')
    
    # Set limits
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    
    # Add legend
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Make the plot look nice
    plt.tight_layout()
    
    # Save to wandb
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    
    # Log to wandb
    if original:
        wandb_run.log({f"plots/original/{dataset}/ece": wandb.Image(img)})
    else:
        wandb_run.log({f"plots/fine-tuned/{dataset}/ece": wandb.Image(img)})
    
    plt.show()

def compute_conf_metrics(y_true, y_confs, number, verbose=True):

    result_matrics = {}
    # ACC
    accuracy = sum(y_true) / len(y_true)
    accuracy2 = sum(y_true) / number
    if verbose:
        print("accuracy: ", accuracy)
        print("accuracy with none: ", accuracy2)
    result_matrics['acc'] = accuracy
    result_matrics['acc2'] = accuracy2

    # use np to test if y_confs are all in [0, 1]
    assert all([x >= 0 and x <= 1 for x in y_confs]), y_confs
    y_confs, y_true = np.array(y_confs), np.array(y_true)
    
    # AUCROC
    roc_auc = roc_auc_score(y_true, y_confs)
    if verbose:
        print("ROC AUC score:", roc_auc)
    result_matrics['auroc'] = roc_auc

    # AUPRC-Positive
    auprc = average_precision_score(y_true, y_confs)
    if verbose:
        print("AUC PRC Positive score:", auprc)
    result_matrics['auprc_p'] = auprc

    # AUPRC-Negative
    auprc = average_precision_score(1- y_true, 1 - y_confs)
    if verbose:
        print("AUC PRC Negative score:", auprc)
    result_matrics['auprc_n'] = auprc
    
    # AURC from https://github.com/IML-DKFZ/fd-shifts/tree/main
    aurc = area_under_risk_coverage_score(y_confs, y_true)
    result_matrics['aurc'] = aurc
    if verbose:
        print("AURC score:", aurc)


    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(np.array(y_confs, dtype=np.float64), np.array(y_true))
    if verbose:
        print("ECE:", ece_score)
    result_matrics['ece'] = ece_score

    return result_matrics