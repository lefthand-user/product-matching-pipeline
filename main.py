# """
# Entry point for the BetterBasket matching project.

# Phase 1:
#     - Load raw datasets for stores A and B
#     - Build simple canonical titles
#     - Initialize the baseline matcher
#     - Run a placeholder matching pipeline

# Later phases will:
#     - produce matches.csv
#     - run full evaluation, etc.
# """

# from src.utils import load_data
# from src.matching_baseline import BaselineMatcher


# def main() -> None:
#     # Paths are relative to project root
#     path_a = "data/grocery_store_a_raw_data.csv"
#     path_b = "data/grocery_store_b_raw_data.csv"

#     df_a, df_b = load_data(path_a, path_b)
#     print(f"Loaded A: {len(df_a)} rows, B: {len(df_b)} rows")

#     matcher = BaselineMatcher(df_a, df_b)
#     matches = matcher.run()

#     print(f"\nPhase 1 baseline produced {len(matches)} matches (expected 0).")
#     print("Pipeline is wired correctly; later phases will implement real matching.")


# if __name__ == "__main__":
#     main()
