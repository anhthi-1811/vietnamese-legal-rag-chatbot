import pandas as pd
import numpy as np
import re
import ast

class ProcessedDataValidator:
    def __init__(self, raw_corpus: pd.DataFrame, processed_corpus: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.raw = raw_corpus
        self.clean = processed_corpus
        self.train = train_df
        self.test = test_df
        self.summary_table = []
        self.all_passed = True

    def _parse_cids(self, val):
        if isinstance(val, list) or isinstance(val, np.ndarray): return list(val)
        if isinstance(val, str):
            try: return ast.literal_eval(val)
            except: return [val]
        return [val]

    def _add_to_summary(self, check_name, expected, actual, passed):
        status = "PASS" if passed else "FAIL"
        self.summary_table.append({
            "Check": check_name, 
            "Expected": expected, 
            "Actual": actual, 
            "Status": status
        })
        if not passed:
            self.all_passed = False

    def validate_integrity(self):
        """Checks row counts, missing CIDs, and duplicates."""
        raw_len = len(self.raw)
        clean_len = len(self.clean)
        self._add_to_summary("Total rows", raw_len, clean_len, raw_len == clean_len)

        clean_unique = self.clean['cid'].nunique()
        self._add_to_summary("Unique CID", raw_len, clean_unique, raw_len == clean_unique)

        missing_cids = self.clean['cid'].isna().sum()
        self._add_to_summary("Missing CID", 0, missing_cids, missing_cids == 0)

        dup_cids = self.clean.duplicated(subset=['cid']).sum()
        self._add_to_summary("Duplicate CID", 0, dup_cids, dup_cids == 0)

    def validate_text_quality(self):
        """Checks for nulls, empty strings, and artifacts."""
        null_texts = self.clean['text_clean'].isna().sum()
        self._add_to_summary("Null text", 0, null_texts, null_texts == 0)

        carriage_returns = self.clean['text_clean'].str.contains(r'\r').sum()
        self._add_to_summary("Carriage returns (\\r)", 0, carriage_returns, carriage_returns == 0)

        zero_width = self.clean['text_clean'].str.contains(r'[\u200B-\u200D\uFEFF]').sum()
        self._add_to_summary("Zero-width characters", 0, zero_width, zero_width == 0)

    def validate_flags(self):
        """Validates the boolean quality flags."""
        short_count = self.clean['is_very_short'].sum()
        self._add_to_summary("Very short docs flag", 34, short_count, short_count == 34)

        long_count = self.clean['needs_local_passage_selection'].sum()
        self._add_to_summary("Long docs flag", 1993, long_count, long_count == 1993)

        punct_count = self.clean['is_punctuation_only'].sum()
        self._add_to_summary("Punctuation-only flag", 7, punct_count, punct_count == 7)

    def validate_benchmark_safety(self):
        """Ensures all gold CIDs exist in the processed corpus."""
        train_cids = set([str(cid) for sublist in self.train['cid'].apply(self._parse_cids) for cid in sublist])
        test_cids = set([str(cid) for sublist in self.test['cid'].apply(self._parse_cids) for cid in sublist])
        all_gold = train_cids.union(test_cids)
        
        processed_cids = set(self.clean['cid'].astype(str).tolist())
        missing = all_gold - processed_cids
        
        coverage = ((len(all_gold) - len(missing)) / len(all_gold)) * 100
        
        self._add_to_summary("Gold CID coverage", "100.0%", f"{coverage:.1f}%", len(missing) == 0)

    def run_all_validations(self):
        """Executes all validation steps and returns the summary."""
        print("Running Integrity Checks...")
        self.validate_integrity()
        
        print("Running Text Quality Checks...")
        self.validate_text_quality()
        
        print("Running Quality Flags Checks...")
        self.validate_flags()
        
        print("Running Benchmark Safety Checks...")
        self.validate_benchmark_safety()
        
        return pd.DataFrame(self.summary_table), self.all_passed 