import argparse
import json
import os
from glob import glob

import numpy as np
import torch

# Allow legacy Lightning/PyTorch checkpoints that store numpy reconstruct/ndarray/dtype variants
try:
    torch.serialization.add_safe_globals(
        [
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            getattr(np, "dtypes", np).__dict__.get("Float32DType", None),
            getattr(np, "dtypes", np).__dict__.get("Float64DType", None),
            np.core.multiarray.scalar,
        ]
    )
except Exception:
    pass

from autoencoder import Autoencoder, split_train_val_test
from discriminator import Discriminator
from geodesic_fm import GeodesicFlowMatching
from off_manifold import make_offmanifold
from train import (
    encode_data,
    eval_wasserstein,
    load_data,
    setup_seed,
    visualize_generated,
    visualize_latent,
    visualize_trajectory,
)


def _load_saved_args(checkpoint_dir):
    args_path = os.path.join(checkpoint_dir, "run_args.json")
    if not os.path.exists(args_path):
        return {}
    with open(args_path, "r") as f:
        return json.load(f)


def _load_splits(checkpoint_dir, x, test_size, val_size):
    splits_path = os.path.join(checkpoint_dir, "data_splits.npz")
    if os.path.exists(splits_path):
        splits = np.load(splits_path)
        return splits["train_idx"], splits["val_idx"], splits["test_idx"]
    return split_train_val_test(x, test_size=test_size, val_size=val_size)


def _find_ckpt(explicit_path, checkpoint_dir, name_pattern, required=True):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    direct = os.path.join(checkpoint_dir, f"{name_pattern}.ckpt")
    if os.path.exists(direct):
        return direct
    matches = glob(os.path.join(checkpoint_dir, "**", f"{name_pattern}*.ckpt"), recursive=True)
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Could not locate {name_pattern} checkpoint. Provide --{name_pattern}_ckpt.")
    return None


def build_models(cfg, device, args):
    ae_ckpt = _find_ckpt(args.autoencoder_ckpt, args.checkpoint_dir, "autoencoder")
    gb_ckpt = _find_ckpt(args.gbmodel_ckpt, args.checkpoint_dir, "gbmodel")
    disc_ckpt = _find_ckpt(
        args.discriminator_ckpt, args.checkpoint_dir, "discriminator", required=False
    )

    ae_model = Autoencoder.load_from_checkpoint(ae_ckpt).to(device)
    disc_model = Discriminator.load_from_checkpoint(disc_ckpt).to(device) if disc_ckpt else None
    ae_model.eval()
    if disc_model:
        disc_model.eval()

    enc_func = lambda x: ae_model.encoder(x, normalize=True)
    ofm = (
        make_offmanifold(enc_func, disc_model, disc_factor=cfg.get("disc_factor", 10.0))
        if disc_model
        else lambda x: enc_func(x)
    )

    gbmodel = GeodesicFlowMatching.load_from_checkpoint(
        gb_ckpt,
        func=ofm,
        encoder=enc_func,
        input_dim=cfg["data_dim"],
        hidden_dim=cfg.get("hidden_dim", 64),
        scale_factor=cfg.get("scale_factor", 1.0),
        embed_t=cfg.get("embed_t", False),
        num_layers=cfg.get("num_layers", 3),
        n_tsteps=cfg.get("n_tsteps", 100),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        flow_weight=cfg.get("flow_weight", 1.0),
        length_weight=cfg.get("length_weight", 1.0),
    ).to(device)
    gbmodel.eval()
    return ae_model, disc_model, gbmodel


def evaluate(args):
    saved = _load_saved_args(args.checkpoint_dir)
    cfg = saved.copy()
    cfg["data_path"] = args.data_path or saved.get("data_path") or "../../data/cite_all_D-100_d-3_pca.npz"
    cfg["plots_save_dir"] = args.plots_save_dir or saved.get("plots_save_dir") or "./plots_minimal"
    cfg["checkpoint_dir"] = args.checkpoint_dir
    cfg["seed"] = args.seed if args.seed is not None else saved.get("seed", 1111)
    cfg["deterministic"] = args.deterministic or saved.get("deterministic", False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    setup_seed(cfg["seed"], deterministic=cfg["deterministic"])
    os.makedirs(cfg["plots_save_dir"], exist_ok=True)

    x, x_dists, labels, _ = load_data(cfg["data_path"])
    cfg["data_dim"] = x.shape[1]

    train_idx, val_idx, test_idx = _load_splits(
        args.checkpoint_dir,
        x,
        test_size=saved.get("test_size", 0.1),
        val_size=saved.get("val_size", 0.1),
    )
    train_x, val_x, test_x = x[train_idx], x[val_idx], x[test_idx]
    train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]

    ae_model, disc_model, gbmodel = build_models(cfg, device, args)

    x_encodings = encode_data(x, ae_model.encoder, device)
    visualize_latent(x_encodings, labels, os.path.join(cfg["plots_save_dir"], "latent_space_eval.png"))

    wd_model = disc_model  # alias for clarity
    wd_model.eval()

    test_start_pts = test_x[test_labels == saved.get("start_group", 0)]
    test_end_pts = test_x[test_labels == saved.get("end_group", 2)]

    ts_eval = torch.linspace(0, 1, saved.get("n_tsteps", 100), device=device)
    gb_trajs = gbmodel.cc(
        torch.tensor(test_start_pts, dtype=torch.float32, device=device),
        torch.tensor(test_end_pts, dtype=torch.float32, device=device),
        ts_eval,
    )
    gb_trajs_enc = encode_data(
        gb_trajs.reshape(-1, gb_trajs.shape[-1]), ae_model.encoder, device
    ).reshape(saved.get("n_tsteps", 100), -1, saved.get("ae_latent_dim", 3))
    start_pts_enc = encode_data(test_start_pts, ae_model.encoder, device)
    end_pts_enc = encode_data(test_end_pts, ae_model.encoder, device)

    visualize_trajectory(
        traj=gb_trajs_enc,
        x_encodings=x_encodings,
        labels=labels,
        save_path=os.path.join(cfg["plots_save_dir"], "geodesic_paths_latent_eval.png"),
        start_pts=start_pts_enc,
        end_pts=end_pts_enc,
    )

    real_idx = np.where(test_labels == saved.get("test_group", 1))[0]
    real_data = test_x[real_idx]
    generated_data = gb_trajs.reshape(-1, gb_trajs.shape[-1]).cpu().numpy()
    w1 = eval_wasserstein(generated_data, real_data)
    with open(os.path.join(cfg["plots_save_dir"], "eval.log"), "a") as f:
        f.write(f"Wasserstein-1 distance (target group {saved.get('test_group', 1)}): {w1}\n")
    visualize_generated(
        generated_data,
        real_data,
        x_encodings,
        labels,
        os.path.join(cfg["plots_save_dir"], f"generated_vs_real_t{saved.get('test_group', 1)}_eval.png"),
    )
    print(f"Eval complete. W1 distance: {w1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate flow-matching minimal checkpoints on CITE data.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_minimal")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path (else use saved run_args).")
    parser.add_argument("--plots_save_dir", type=str, default=None)
    parser.add_argument("--autoencoder_ckpt", type=str, default=None)
    parser.add_argument("--discriminator_ckpt", type=str, default=None)
    parser.add_argument("--gbmodel_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
