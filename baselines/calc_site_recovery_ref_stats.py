import sys
import os
import math
import argparse
import torch
import numpy as np
import glob
from rdkit import Chem
import tqdm
from neuralplexer.util.pdb3d import get_lddt_bs
from af_common.protein import Protein, from_prediction, to_pdb
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import pandas as pd
import re
from subprocess import check_output
import multiprocessing as mp
from subprocess import check_output, run


def main(dummy=False):
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(
        "datasets/pdbbind-raw/timesplit_no_lig_overlap_test_0830_chain_seq.csv"
    )
    res = []
    for _, row in tqdm.tqdm(df.iterrows()):
        try:
            sample_id = row["id"]
            pdb_id = row["pdb_id"].lower()
            chain_id = row["chain"]
            colabfold_path = glob.glob(
                f"colabfold_predictions/pdbbind_test_set/{sample_id}_relaxed_rank_1_*.pdb"
            )[0]
            ref_pdb_path = f"datasets/pdbbind-raw/all/{pdb_id}/{pdb_id}_protein.pdb"
            ref_lig_path = f"datasets/pdbbind-raw/all/{pdb_id}/{pdb_id}_ligand.sdf"
            chain_path = f"datasets/pdbbind-raw/pdb_selchain/{sample_id}.pdb"
            # See: https://github.com/haddocking/pdb-tools/issues/101
            with open(chain_path, "w") as f:
                run(
                    f"pdb_selchain -{chain_id} {ref_pdb_path} | pdb_delhetatm | pdb_tidy -strict | grep -v ANISOU",
                    shell=True,
                    stdout=f,
                )
            tm_scores = []
            rmsds = []
            lddt_bs = get_lddt_bs(colabfold_path, chain_path, ref_lig_path)
            tm_score = None
            align_out_path = f"colabfold_predictions/TMaligned/{sample_id}.sup"
            ret = check_output(
                ["TMalign", colabfold_path, chain_path, "-o", align_out_path]
            )
            align_save_path = (
                f"colabfold_predictions/TMaligned/{sample_id}_AF2_aligned.pdb"
            )
            with open(align_save_path, "w") as f:
                run(
                    f"pdb_selchain -A {align_out_path} | pdb_delhetatm | pdb_tidy -strict | grep -v ANISOU",
                    shell=True,
                    stdout=f,
                )
            for line in str(ret).split("\\n"):
                if re.match(r"^Aligned", line):
                    rmsd = float(line.split()[4][:-1])  # Extract the value
                if re.match(r"^TM-score=", line):
                    tm_score = line.split()[1]  # Extract the value
            tm_scores.append(tm_score)
            rmsds.append(rmsd)
            res.append([sample_id] + tm_scores + rmsds + [lddt_bs])
        except Exception as e:
            print(e)
            continue
        res_df = pd.DataFrame(
            columns=["sample_id", "TMScore", "RMSD", "lDDT-BS"],
            data=res,
        )
        res_df.to_csv(args.csv_path)


if __name__ == "__main__":
    sys.exit(main())
