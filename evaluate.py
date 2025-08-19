# evaluate.py
import os
import json
import glob
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import get_evaluation_config, get_app_config


class Evaluator:
    """
    Evaluates generated video summaries against ground-truth captions
    using BLEU, METEOR, and ROUGE metrics.
    Also provides exportable metrics and plots.
    """
    def __init__(self, ground_truth_path, results_root=None, output_dir=None):
        # Get config
        eval_config = get_evaluation_config()
        app_config = get_app_config()
        
        self.ground_truth_path = ground_truth_path
        self.results_root = results_root or app_config.results_dir
        self.output_dir = output_dir or eval_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics = {}
        for metric_name in eval_config.metrics:
            self.metrics[metric_name] = evaluate.load(metric_name)

    def load_ground_truth(self):
        """Load ground-truth video captions from JSON."""
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        references = {}
        for entry in data["sentences"]:
            vid = entry["video_id"]
            references.setdefault(vid, []).append(entry["caption"].strip().lower())
        return references

    def load_generated_summaries(self):
        """Load all predicted summaries from results folder."""
        summaries = {}
        pattern = os.path.join(self.results_root, "*", "*", "captions", "*_captions.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                video_id = data.get("video_id")
                if video_id and "summary" in data:
                    summaries[video_id] = data["summary"].strip().lower()
            except Exception as e:
                print(f"⚠️ Error reading {filepath}: {e}")
        return summaries

    def evaluate(self, references, predictions):
        """
        Compute BLEU, METEOR, and ROUGE scores per video.

        Parameters:
            references (dict): {video_id: [reference captions]}
            predictions (dict): {video_id: predicted summary}

        Returns:
            pd.DataFrame: metrics per video
        """
        results = []
        for vid, pred in tqdm(predictions.items(), desc="Evaluating"):
            if vid not in references:
                continue
            refs = references[vid]
            scores = {
                "video_id": vid,
                "prediction": pred,
                "references": refs,
                "bleu": self.metrics["bleu"].compute(predictions=[pred], references=[refs])["bleu"],
                "meteor": self.metrics["meteor"].compute(predictions=[pred], references=[refs])["meteor"],
                "rouge": self.metrics["rouge"].compute(predictions=[pred], references=[refs])["rougeL"]
            }
            results.append(scores)
        return pd.DataFrame(results)

    def plot_metrics(self, df):
        """Save line plot of evaluation metrics per video."""
        plt.figure(figsize=(12, 6))
        x = range(len(df))
        plt.plot(x, df["bleu"], marker="o", label="BLEU")
        plt.plot(x, df["meteor"], marker="s", label="METEOR")
        plt.plot(x, df["rouge"], marker="^", label="ROUGE-L")
        plt.xticks(x, df["video_id"], rotation=45, ha="right")
        plt.ylabel("Score")
        plt.xlabel("Video ID")
        plt.title("Evaluation Metrics per Video")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "metrics_plot.png")
        plt.savefig(out_path)
        plt.close()
        print(f"📊 Plot saved to: {out_path}")

    def export_results(self, df):
        """Save raw scores and averages to CSV and TXT."""
        csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Evaluation results saved to: {csv_path}")

        averages = df[["bleu", "meteor", "rouge"]].mean()
        avg_path = os.path.join(self.output_dir, "averages.txt")
        with open(avg_path, "w", encoding="utf-8") as f:
            f.write("📊 Averages:\n")
            f.write(averages.to_string())
        print(f"✅ Averages saved to: {avg_path}")
        print("\n📊 Averages:\n", averages)


if __name__ == "__main__":
    evaluator = Evaluator(
        ground_truth_path="data/captions/train_val_videodatainfo.json"
    )

    print("🔹 Loading ground-truth captions...")
    refs = evaluator.load_ground_truth()

    print("🔹 Loading generated summaries...")
    preds = evaluator.load_generated_summaries()

    print(f"🔹 Evaluating {len(preds)} summaries...")
    df = evaluator.evaluate(refs, preds)

    evaluator.export_results(df)
    evaluator.plot_metrics(df)
