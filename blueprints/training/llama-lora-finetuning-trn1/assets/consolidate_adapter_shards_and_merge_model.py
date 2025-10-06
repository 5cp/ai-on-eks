from optimum.neuron.models.training.checkpointing import (
    consolidate_model_parallel_checkpoints_to_unified_checkpoint,
)
from transformers import AutoModel
from argparse import ArgumentParser
from shutil import copyfile
import os
import peft

parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    help="Source checkpoint directory containing sharded adapter checkpoint files",
    required=True,
)
parser.add_argument(
    "-o",
    "--output_dir",
    help="Destination directory for final merged model (adapters merged into base model)",
    required=True,
)
args = parser.parse_args()

consolidated_ckpt_dir = os.path.join(args.input_dir, "adapter_default")

print("Consolidating the adapter shards into a PEFT-compatible checkpoint")
consolidate_model_parallel_checkpoints_to_unified_checkpoint(
    args.input_dir, consolidated_ckpt_dir
)

# Load AutoPeftModel using the consolidated PEFT checkpoint
peft_model = peft.AutoPeftModelForCausalLM.from_pretrained(consolidated_ckpt_dir)

print(f"Merging adapter weights into base model and saving new pretrained model to: {args.output_dir}")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(args.output_dir)

print("Loading the pretrained model and printing model")
model = AutoModel.from_pretrained(args.output_dir)
print(model)
