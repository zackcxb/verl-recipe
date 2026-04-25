"""Minimal dataset for the DeepEyes gateway recipe.

Produces ``raw_prompt`` and reward-related fields only.
It does not perform tokenization or vision processing.
"""

from __future__ import annotations

import logging

import torch

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class DeepEyesGatewayDataset(RLHFDataset):
    """Thin dataset that leaves prompt encoding and vision extraction to the gateway."""

    def maybe_filter_out_long_prompts(self, dataframe=None):
        """Skip base prompt filtering because phase 1 must not tokenize or preprocess vision here."""
        return self.dataframe if dataframe is None else dataframe

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        raw_messages = self._build_messages(row_dict, key=self.prompt_key)
        row_dict["raw_prompt"] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You can call functions to assist "
                    "with the user query. Important: You must call only one function "
                    "at a time."
                ),
            },
            {
                "role": "user",
                "content": raw_messages[1]["content"] if len(raw_messages) > 1 else raw_messages[0].get("content", ""),
            },
        ]
        if self.negative_prompt_key in row_dict:
            row_dict["raw_negative_prompt"] = self._build_messages(row_dict, key=self.negative_prompt_key)

        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        extra_info = row_dict["extra_info"]
        index = extra_info.get("index", 0)
        tools_kwargs = extra_info.get("tools_kwargs", {})
        need_tools_kwargs = extra_info.get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict.get("data_source"))
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs

        return row_dict
