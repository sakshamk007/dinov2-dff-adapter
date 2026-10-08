#!/usr/bin/env python3
import argparse
import yaml
import torch
from train import evaluate, get_loader
from dff_adapter.model import DFFAdapterDetector

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True); ap.add_argument("--manifest")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); device = torch.device("cuda")
    model = DFFAdapterDetector(cfg["model_name"], cfg["num_forgery_types"], cfg["adapter"]).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device)["model"])
    print(evaluate(model, get_loader(a.manifest or cfg["data"]["test_manifest"], cfg, False), device))
if __name__ == "__main__": main()
