import time
from pathlib import Path
import pandas as pd
from parallax.data.contracts import load_business_records_df
from parallax.preprocessing.normalizer import widen_records_df
from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker

def main():
    print("Loading data...")
    data_dir = Path("data/medium_split_200k")
    s1_df = load_business_records_df(data_dir / "train_source1.tsv").head(500)
    s2_df = load_business_records_df(data_dir / "train_source2.tsv")
    s3_df = load_business_records_df(data_dir / "train_source3.tsv")
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    print("Widening...")
    s1_wide = widen_records_df(s1_df)
    target_wide = widen_records_df(target_df)
    
    print("Running Blocker with Channel D at K=5")
    blocker = DualChannelTFIDFBlocker(
        name_top_k=35,
        addr_top_k=25,
        translit_top_k=5,
        name_min_sim=0.15,
        addr_min_sim=0.20,
        translit_min_sim=0.30,
        batch_size=2000,
        show_progress=True,
    )
    
    candidates = blocker.generate_candidates(s1_wide, target_wide)
    total_pairs = sum(len(c) for c in candidates.values())
    print(f"Total unioned pairs from K=5 blocker: {total_pairs}")
    print(f"Average candidates per S1: {total_pairs / len(s1_wide):.2f}")
    
if __name__ == '__main__':
    main()
