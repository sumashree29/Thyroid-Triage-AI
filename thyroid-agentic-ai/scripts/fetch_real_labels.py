import pandas as pd
from pathlib import Path

def main():
    print("Fetching real UCI Thyroid Disease (sick) diagnostic labels...")
    url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.data'
    url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.test'

    # The dataset has features as the first 29 columns and the target as the 30th column
    df_data = pd.read_csv(url_data, header=None)
    df_test = pd.read_csv(url_test, header=None)
    
    # Concatenate to get the full 3772 patients in the exact original order
    df_all = pd.concat([df_data, df_test], ignore_index=True)
    
    # The last column contains strings like "negative.|3733" or "sick.|2424"
    raw_labels = df_all.iloc[:, -1].str.split('.\|').str[0]
    
    print("Target value counts:")
    print(raw_labels.value_counts())
    
    def map_to_risk(raw_label: str) -> int:
        # We only have negative and sick in this specific dataset split
        val = str(raw_label).strip().lower()
        if "sick" in val:
            return 1 # High Risk
        else:
            return 0 # Low Risk
            
    y_mapped = raw_labels.apply(map_to_risk)
    
    # Load the original features that the pipeline expects
    X_path = Path("data/raw/Thyroid_Data.csv")
    if not X_path.exists():
        X_path = Path("../data/raw/Thyroid_Data.csv")
    
    merged = pd.read_csv(X_path)
    
    # In case there's an old target or derive column, drop it (though Thyroid_Data.csv doesn't have one)
    if "target" in merged.columns:
        merged = merged.drop(columns=["target"])
        
    merged["target"] = y_mapped
    
    out_path = Path("data/raw/Thyroid_Data_labeled.csv")
    merged.to_csv(out_path, index=False)
    
    print(f"Saved {out_path} with real diagnostic labels")
    print("Mapped target value counts:")
    print(merged["target"].value_counts())

if __name__ == "__main__":
    main()
