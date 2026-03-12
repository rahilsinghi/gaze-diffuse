"""Gaze fixation duration predictor (Experiment 2).

Fine-tunes BERT-base-uncased on the GECO eye-tracking corpus to predict
per-word mean fixation duration from context. Used as the guidance signal
in both AR baseline and GazeDiffuse experiments.

Architecture (Sauberli et al.):
    Input:  [CLS] left_5_tokens [SEP] target_token [SEP] right_5_tokens [SEP]
    Output: Predicted mean fixation duration (ms) via [CLS] pooling + linear head

Usage:
    python -m src.gaze_predictor --data_dir data/geco --epochs 3 --output_dir checkpoints/gaze_predictor
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from src.data.geco import (
    GazeDataConfig,
    GazeDataset,
    create_subject_cv_splits,
    extract_gaze_examples,
    load_geco_corpus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GazePredictorConfig:
    """Configuration for the gaze predictor model."""

    bert_model: str = "bert-base-uncased"
    hidden_size: int = 768
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    max_seq_length: int = 64
    warmup_ratio: float = 0.1


class GazePredictor(nn.Module):
    """BERT-based gaze fixation duration predictor.

    Fine-tunes BERT-base on regression: predict mean fixation duration
    for a target word given its surrounding context.
    """

    def __init__(self, config: GazePredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.bert_model)
        self.dropout = nn.Dropout(config.dropout)
        self.regressor = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict fixation duration from contextualized input.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask

        Returns:
            [batch] predicted fixation durations (normalized scale)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] pooling
        cls_output = self.dropout(cls_output)
        prediction = self.regressor(cls_output).squeeze(-1)  # [batch]
        return prediction

    def score_tokens(
        self,
        token_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        context_window: int = 5,
    ) -> torch.Tensor:
        """Score each token position in a sequence by predicted fixation.

        Used during guided generation: for each position, construct a
        context window and predict how long a reader would fixate.

        Args:
            token_ids: [seq_len] token IDs of the current sequence
            tokenizer: Tokenizer for decoding tokens to words
            context_window: Number of context tokens on each side

        Returns:
            [seq_len] predicted fixation durations for each position
        """
        device = next(self.parameters()).device
        seq_len = len(token_ids)
        scores = torch.zeros(seq_len, device=device)

        # Decode tokens to build context windows
        tokens = [tokenizer.decode([tid]).strip() for tid in token_ids]

        batch_inputs = []
        valid_positions = []

        for i in range(seq_len):
            # Skip special tokens
            if token_ids[i].item() in {
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
            }:
                continue

            left_start = max(0, i - context_window)
            right_end = min(seq_len, i + context_window + 1)

            left_ctx = " ".join(tokens[left_start:i])
            right_ctx = " ".join(tokens[i + 1 : right_end])
            target = tokens[i]

            encoding = tokenizer(
                text=f"{left_ctx} [SEP] {target}",
                text_pair=right_ctx,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            batch_inputs.append(encoding)
            valid_positions.append(i)

        if not batch_inputs:
            return scores

        # Batch all positions together
        input_ids = torch.cat(
            [inp["input_ids"] for inp in batch_inputs], dim=0
        ).to(device)
        attention_mask = torch.cat(
            [inp["attention_mask"] for inp in batch_inputs], dim=0
        ).to(device)

        with torch.no_grad():
            predictions = self.forward(input_ids, attention_mask)

        for pos, pred in zip(valid_positions, predictions):
            scores[pos] = pred

        return scores

    def score_vocabulary(
        self,
        sequence: torch.Tensor,
        position: int,
        vocab_candidates: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        context_window: int = 5,
    ) -> torch.Tensor:
        """Score vocabulary candidates at a specific position.

        For each candidate token, substitute it into the position and
        predict fixation duration. Used in AR-style per-token guidance.

        Args:
            sequence: [seq_len] current token IDs
            position: Index to evaluate candidates at
            vocab_candidates: [n_candidates] candidate token IDs
            tokenizer: Tokenizer for decoding
            context_window: Context window size

        Returns:
            [n_candidates] predicted fixation for each candidate
        """
        device = next(self.parameters()).device
        n_candidates = len(vocab_candidates)
        tokens = [tokenizer.decode([tid]).strip() for tid in sequence]

        left_start = max(0, position - context_window)
        left_ctx = " ".join(tokens[left_start:position])
        right_end = min(len(sequence), position + context_window + 1)
        right_ctx = " ".join(tokens[position + 1 : right_end])

        # Build batch: one example per candidate
        candidate_words = [
            tokenizer.decode([cid]).strip() for cid in vocab_candidates
        ]

        encodings = tokenizer(
            [f"{left_ctx} [SEP] {w}" for w in candidate_words],
            [right_ctx] * n_candidates,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            predictions = self.forward(input_ids, attention_mask)

        return predictions


def train_gaze_predictor(
    config: GazePredictorConfig,
    data_config: GazeDataConfig,
    output_dir: str | Path,
) -> dict[str, float]:
    """Train the gaze predictor on GECO corpus.

    Returns:
        Dictionary with training metrics (loss, spearman_r).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Load data
    df = load_geco_corpus(data_config.data_dir)
    examples = extract_gaze_examples(df, data_config)
    logger.info("Loaded %d gaze examples", len(examples))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)

    # Subject-level CV: use first fold for train/val split
    splits = list(create_subject_cv_splits(examples, n_folds=5))
    train_examples, val_examples = splits[0]

    train_dataset = GazeDataset(
        train_examples, tokenizer, config.max_seq_length, normalize=True
    )
    val_dataset = GazeDataset(
        val_examples, tokenizer, config.max_seq_length, normalize=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    # Model
    model = GazePredictor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    # Linear warmup + decay scheduler
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_spearman = -1.0
    metrics: dict[str, float] = {}

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        all_preds: list[float] = []
        all_labels: list[float] = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, labels)
                val_loss += loss.item()

                all_preds.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        spearman_r, _ = spearmanr(all_preds, all_labels)

        logger.info(
            "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f, spearman_r: %.4f",
            epoch + 1,
            config.epochs,
            avg_train_loss,
            avg_val_loss,
            spearman_r,
        )

        if spearman_r > best_spearman:
            best_spearman = spearman_r
            checkpoint_path = output_dir / "gaze_predictor_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "normalization": train_dataset.get_normalization_stats(),
                    "spearman_r": spearman_r,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            logger.info("Saved best checkpoint (r=%.4f) to %s", spearman_r, checkpoint_path)

    metrics = {
        "best_spearman_r": best_spearman,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
    }

    # Save final checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "normalization": train_dataset.get_normalization_stats(),
            "metrics": metrics,
        },
        output_dir / "gaze_predictor_final.pt",
    )

    return metrics


def load_trained_predictor(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[GazePredictor, tuple[float, float]]:
    """Load a trained gaze predictor from checkpoint.

    Returns:
        (model, (mean, std)) — the model and normalization stats.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GazePredictor(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    norm_stats = checkpoint.get("normalization", (0.0, 1.0))
    return model, norm_stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train gaze fixation predictor")
    parser.add_argument("--data_dir", type=str, default="data/geco")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gaze_predictor")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    pred_config = GazePredictorConfig(
        bert_model=args.bert_model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    data_cfg = GazeDataConfig(data_dir=args.data_dir)

    results = train_gaze_predictor(pred_config, data_cfg, args.output_dir)
    logger.info("Training complete. Results: %s", results)
