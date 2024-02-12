import pandas as pd
import tqdm
import numpy as np

df = pd.read_csv("PDBBind2019+bychain_AF2selected_test.csv")

res = []
for sample_id in tqdm.tqdm(df["sample_id"]):
    try:
        rmsds = np.loadtxt(
            f"PDBBind2019+bychain_AF2selected_test033023/{sample_id}/lig_rmsd.summary"
        )
        lddts = np.loadtxt(
            f"PDBBind2019+bychain_AF2selected_test033023/{sample_id}/rec_lddtbs.summary"
        )
        res.append([sample_id] + lddts.tolist() + rmsds.tolist() + [0] * 32)
    except:
        continue

res_df = pd.DataFrame(
    columns=["sample_id"]
    + [f"lDDT-BS_{i}" for i in range(32)]
    + [f"lig_rmsd_{i}" for i in range(32)]
    + [f"clash_{i}" for i in range(32)],
    data=res,
)
res_df.to_csv(f"benchmarking_summary_033023.csv")
