# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Daniel Rosehill for Meorav TTS (Hebrish), 2026-10-21/22: a copy of finetuning/sft_12hz.py at
# QwenLM/Qwen3-TTS 022e286b98 (2026-03-17), with fixes upstream has not merged (checked 2026-10-21). Each
# change is marked "VC1026" (the working name of the project when the fixes were written). Why:
# docs/training-plan.md.
#   1. Label shift: the talker got inputs[:-1] with labels[1:] and HF's causal loss shifted again, so it
#      learned frame t+2 (QwenLM/Qwen3-TTS#371, #179; fix as PR #278). Full sequence + unshifted labels now.
#   2. Sub-talker input: it took the hidden state at the frame's own position, which has already seen that
#      frame's codes (#337). Now the previous position, as at inference (PR #278).
#   3. Sub-talker loss: forward_finetune's labels are already aligned and the causal loss shifted them
#      once more (PR #278 patches the package). Computed here from its logits instead.
#   4. text_projection was skipped in training but applied at inference (#120, PR #188/#336).
#   5. bf16 weights with plain AdamW round most updates away at LR 2e-6..2e-5: fp32 master weights and
#      optimizer state, bf16 autocast (--fp32_master, default on). Checkpoints are written in bf16.
#   6. --save_every N: a checkpoint every N epochs and after the last (each is ~4.5 GB).
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--save_every", type=int, default=1)  # VC1026 (6)
    parser.add_argument("--fp32_master", type=int, default=1)  # VC1026 (5)
    args = parser.parse_args()

    # VC1026 (7): upstream passes log_with="tensorboard" with no logging_dir, which accelerate 1.12 rejects
    # (ValueError, first Modal run 2026-10-22); no tracker is ever initialised, so loss goes to train.log only.
    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32 if args.fp32_master else torch.bfloat16,  # VC1026 (5)
        attn_implementation="sdpa",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.text_projection(  # VC1026 (4)
                    model.talker.model.text_embedding(input_text_ids)) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # VC1026 (1): labels are input-aligned (dataset.py); the causal loss does the one shift.
                outputs = model.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True
                )

                # VC1026 (2): frame k's codes are predicted from the hidden state one position earlier.
                hidden_states = outputs.hidden_states[0][-1][:, :-1, :]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, _ = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                # VC1026 (3): logits[:, i] already predict code i+1; no further shift.
                sub_talker_loss = torch.nn.functional.cross_entropy(
                    sub_talker_logits.reshape(-1, sub_talker_logits.shape[-1]).float(),
                    talker_codec_ids[:, 1:].reshape(-1))

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process and ((epoch + 1) % args.save_every == 0 or epoch == num_epochs - 1):  # VC1026 (6)
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: (v.detach().to("cpu").to(torch.bfloat16) if v.is_floating_point() else v.detach().to("cpu"))  # VC1026 (5)
                          for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
