"""Dataset loading and preprocessing utilities."""

from dataclasses import dataclass
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


@dataclass
class DataConfig:
    """Configuration for data loading."""

    dataset: str = "roneneldan/TinyStories"
    subset: str | None = None  # Dataset subset/config (e.g., "sample-10BT" for FineWeb)
    tokenizer: str = "gpt2"
    max_seq_len: int = 512
    num_workers: int = 4


class TokenizedDataset(IterableDataset):
    """An iterable dataset that tokenizes text on-the-fly."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer_name: str,
        max_seq_len: int,
        seed: int = 42,
        subset: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.max_seq_len = max_seq_len
        self.seed = seed

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Load dataset in streaming mode for memory efficiency
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10000)

        # Buffer for accumulating tokens
        token_buffer: list[int] = []

        for example in dataset:
            # Get text from the example
            text = example.get("text", "")
            if not text:
                continue

            # Tokenize and add to buffer
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)
            token_buffer.append(self.tokenizer.eos_token_id)

            # Yield chunks of max_seq_len + 1 (for input/target split)
            while len(token_buffer) >= self.max_seq_len + 1:
                chunk = token_buffer[: self.max_seq_len + 1]
                token_buffer = token_buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


class FiniteTokenizedDataset(torch.utils.data.Dataset):
    """A finite dataset for validation/test that loads all data into memory."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer_name: str,
        max_seq_len: int,
        max_examples: int = 10000,
        seed: int = 42,
        subset: str | None = None,
    ):
        self.max_seq_len = max_seq_len
        self.max_examples = max_examples

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            streaming=True,
        )
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)

        # Tokenize and chunk
        self.examples: list[dict[str, torch.Tensor]] = []
        token_buffer: list[int] = []

        for example in dataset:
            if len(self.examples) >= max_examples:
                break

            text = example.get("text", "")
            if not text:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)
            token_buffer.append(tokenizer.eos_token_id)

            while len(token_buffer) >= max_seq_len + 1 and len(self.examples) < max_examples:
                chunk = token_buffer[: max_seq_len + 1]
                token_buffer = token_buffer[max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def create_dataloaders(
    config: DataConfig,
    batch_size: int,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        config: Data configuration
        batch_size: Batch size for all dataloaders
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Check if this dataset has a validation split
    # Most large datasets (FineWeb, OpenWebText, etc.) only have train split
    has_validation_split = config.dataset in ["roneneldan/TinyStories"]

    # Training dataset (streaming/infinite)
    train_dataset = TokenizedDataset(
        dataset_name=config.dataset,
        split="train",
        tokenizer_name=config.tokenizer,
        max_seq_len=config.max_seq_len,
        seed=seed,
        subset=config.subset,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Validation dataset (finite, loaded into memory)
    # For datasets without validation split, we use a portion of train with different seed
    val_split = "validation" if has_validation_split else "train"
    val_dataset = FiniteTokenizedDataset(
        dataset_name=config.dataset,
        split=val_split,
        tokenizer_name=config.tokenizer,
        max_seq_len=config.max_seq_len,
        max_examples=5000,
        seed=seed + 500 if not has_validation_split else seed,  # Different seed for train-based val
        subset=config.subset,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No multiprocessing for validation
        pin_memory=True,
    )

    # Test dataset (finite, loaded into memory)
    # Use different seed to get different examples
    test_dataset = FiniteTokenizedDataset(
        dataset_name=config.dataset,
        split=val_split,
        tokenizer_name=config.tokenizer,
        max_seq_len=config.max_seq_len,
        max_examples=5000,
        seed=seed + 1000,  # Different seed to get different examples
        subset=config.subset,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_vocab_size(tokenizer_name: str = "gpt2") -> int:
    """Get the vocabulary size for a tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer.vocab_size
