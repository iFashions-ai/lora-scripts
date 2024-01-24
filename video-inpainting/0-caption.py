from typing import List
from tqdm import tqdm
from pathlib import Path
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from argparse import ArgumentParser


def parse_args(argv: List[str] = None):
    parser = ArgumentParser()
    parser.add_argument(
        "src_dir", type=str, default="/data/video-dataset/VSPW/data/*/origin"
    )
    return parser.parse_args(argv)


args = parse_args()
src_dir = Path(args.src_dir)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
    model_type="vicuna7b",
    is_eval=True,
    device=device,
)


def get_caption(filename: str):
    image = Image.open(filename).convert("RGB")
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})[0]
    return caption


# Generate captions for all frames
images = sorted(src_dir.rglob("*.jpg"), key=lambda p: (p.parents[1], p.name))
for image_file in tqdm(images):
    outputfile = image_file.parents[1] / "caption" / image_file.with_suffix(".txt").name
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    if outputfile.exists():
        continue
    caption = get_caption(image_file)
    with open(outputfile, "w") as f:
        f.write(caption)
