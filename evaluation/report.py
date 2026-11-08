import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.evaluation.experiment_output import OutputSchema, write_csv_with_schema

class ExperimentReport:
    """
    Aggregates metrics and generates reports.
    """
    
    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.results: List[Dict[str, Any]] = []
        
    def add_result(self, metrics: Dict[str, Any]):
        """Add a single experiment run result."""
        metrics['timestamp'] = datetime.now().isoformat()
        self.results.append(metrics)
        self._print_result(metrics)
        
    def _print_result(self, m: Dict[str, Any]):
        """Print a single result row nicely."""
        print("-" * 60)
        print(f"Method: {m.get('method', 'N/A')}")
        print(f"Budget: {m.get('budget', 'N/A')}")
        print(f"Status: {m.get('status', 'N/A')}")
        print(f"Time:   {m.get('time_ms', 0):.2f} ms")
        
        # Security Metrics
        if 'entropy' in m:
            print(f"Entropy:{m.get('entropy', 0):.4f}")
        if 'npcr' in m:
            print(f"NPCR:   {m.get('npcr', 0):.4f}%")
        if 'uaci' in m:
            print(f"UACI:   {m.get('uaci', 0):.4f}%")
        if 'plain_cipher_corr' in m:
            print(f"P-C Corr: {m.get('plain_cipher_corr', 0):.4f}")
        if 'adj_corr_h' in m:
             print(f"Adj Corr (H/V/D): {m.get('adj_corr_h', 0):.4f} / {m.get('adj_corr_v', 0):.4f} / {m.get('adj_corr_d', 0):.4f}")

        utilization = m.get('budget_utilization', 'N/A')
        if isinstance(utilization, float):
            print(f"Budget Utilization: {utilization:.2f}%")
        elif utilization != 'N/A':
            print(f"Budget Utilization: {utilization}")
            
        if 'high_ratio' in m:
            print("Distribution:")
            print(f"  High:   {m.get('high_ratio', 0)*100:.1f}% ({m.get('high_count', 0)})")
            print(f"  Medium: {m.get('medium_ratio', 0)*100:.1f}% ({m.get('medium_count', 0)})")
            print(f"  Low:    {m.get('low_ratio', 0)*100:.1f}% ({m.get('low_count', 0)})")
            print(f"  None:   {m.get('none_ratio', 0)*100:.1f}% ({m.get('none_count', 0)})")
        print("-" * 60)
        
    def save_csv(self, path: str):
        """Save results to CSV with standardized headers."""
        if not self.results:
            return
            
        # Define standard columns to ensure consistency across experiments
        standard_cols = [
            "method", "mode", "budget", "time_ms", 
            "budget_used", "budget_utilization", "status",
            "high_count", "medium_count", "low_count", "none_count",
            "high_ratio", "medium_ratio", "low_ratio", "none_ratio",
            "timestamp"
        ]
        
        # Add any extra columns present in results (e.g. security metrics, deltas, alpha)
        all_keys = set()
        for r in self.results:
            all_keys.update(r.keys())
            
        extra_cols = [k for k in all_keys if k not in standard_cols]
        # Sort extra cols for stability
        extra_cols.sort()
        
        # Final fieldnames
        fieldnames = standard_cols + extra_cols
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        schema = OutputSchema(
            name=f"{self.name}_report",
            fieldnames=tuple(fieldnames),
            required_fields=tuple(standard_cols),
        )
        write_csv_with_schema(
            path,
            self.results,
            schema=schema,
            mode="overwrite",
            strict=True,
        )
        print(f"Report saved to {path}")
            
    def print_summary(self):
        """Print summary stats."""
        print(f"\n=== Experiment Summary: {self.name} ===")
        if not self.results:
            print("No results.")
            return
            
        # Basic averages
        avg_metrics = {}
        count = len(self.results)
        
        for key in self.results[0].keys():
            if isinstance(self.results[0][key], (int, float)) and key != 'timestamp':
                try:
                    # Only sum numeric values
                    valid_values = [r[key] for r in self.results if isinstance(r[key], (int, float))]
                    if valid_values:
                        avg_metrics[key] = sum(valid_values) / len(valid_values)
                except:
                    pass
                
        for k, v in avg_metrics.items():
            print(f"Avg {k}: {v:.4f}")
            
    def print_table(self):
        """Print a comparative table of results."""
        if not self.results:
            return
            
        # Define columns to show - Concise table for terminal
        # Prioritize security metrics if present
        cols = ["method", "budget", "entropy", "npcr", "uaci", "plain_cipher_corr", "time_ms"]
        headers = ["Method", "Budget", "Entropy", "NPCR", "UACI", "P-C Corr", "Time"]
        
        # Check if we have these columns in the first result, if not fallback to basic
        if 'npcr' not in self.results[0]:
             cols = ["method", "budget", "time_ms", "entropy", "budget_utilization", "status"]
             headers = ["Method", "Budget", "Time(ms)", "Entropy", "Util(%)", "Status"]

        # Calculate widths
        widths = [len(h) for h in headers]
        
        # First pass to determine column widths
        formatted_rows = []
        for r in self.results:
            row_vals = []
            for i, c in enumerate(cols):
                raw_val = r.get(c, "")
                val_str = str(raw_val)
                
                # Format specific columns
                if isinstance(raw_val, float):
                    if "ratio" in c: 
                        val_str = f"{raw_val*100:.1f}%"
                    elif "entropy" in c or "corr" in c: 
                        val_str = f"{raw_val:.4f}"
                    elif "npcr" in c or "uaci" in c:
                        val_str = f"{raw_val:.4f}"
                    elif "time" in c: 
                        val_str = f"{raw_val:.1f}"
                    elif "utilization" in c:
                        val_str = f"{raw_val:.1f}"
                
                # Update width
                widths[i] = max(widths[i], len(val_str))
                row_vals.append(val_str)
            formatted_rows.append(row_vals)
                
        # Print
        print("\n" + "=" * (sum(widths) + len(cols)*3 + 1))
        
        # Header
        header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
        print(header_row)
        
        # Separator
        print("|-" + "-|-".join("-" * w for w in widths) + "-|")
        
        # Data rows
        for row_vals in formatted_rows:
            print("| " + " | ".join(f"{val:<{widths[i]}}" for i, val in enumerate(row_vals)) + " |")
            
        print("=" * (sum(widths) + len(cols)*3 + 1) + "\n")

    def print_comparison_table(self, group_by: str = "budget"):
        """
        Print a grouped comparison table.
        Designed for comparing 'rule_based' vs 'cnn' vs 'fusion' rows grouped by 'budget'.
        """
        if not self.results:
            return

        # Identify unique groups
        groups = sorted(list(set(r.get(group_by) for r in self.results if r.get(group_by) is not None)))
        
        print(f"\n=== Comparison Table (Grouped by {group_by}) ===")
        
        # Define columns
        cols = ["mode", "status", "high_ratio", "medium_ratio", "low_ratio", "delta_high_ratio_vs_rule"]
        headers = ["Mode", "Status", "High %", "Med %", "Low %", "Δ High"]
        
        # Calculate widths
        widths = [len(h) for h in headers]
        # Pre-scan for widths
        for r in self.results:
            for i, c in enumerate(cols):
                val = r.get(c, "")
                if isinstance(val, float):
                    if "ratio" in c: val = f"{val*100:.1f}%"
                    elif "delta" in c: val = f"{val*100:+.1f}%"
                widths[i] = max(widths[i], len(str(val)))
                
        # Print
        row_len = sum(widths) + len(cols)*3 + 1
        print("=" * row_len)
        print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |")
        print("|-" + "-|-".join("-" * w for w in widths) + "-|")
        
        for g in groups:
            print(f"| {group_by}: {g:<{row_len - len(group_by) - 5}} |") # Section header
            
            # Get rows for this group
            group_rows = [r for r in self.results if r.get(group_by) == g]
            
            # Sort order: rule_based, fusion, cnn_classifier
            def sort_key(x):
                m = x.get('mode', '')
                if m == 'rule_based': return 0
                if m == 'fusion': return 1
                if m == 'cnn_classifier': return 2
                return 3
            group_rows.sort(key=sort_key)
            
            for r in group_rows:
                vals = []
                for i, c in enumerate(cols):
                    val = r.get(c, "")
                    if isinstance(r.get(c), float):
                        if "ratio" in c and "delta" not in c: val = f"{r[c]*100:.1f}%"
                        elif "delta" in c: val = f"{r[c]*100:+.1f}%"
                    elif val == "":
                         val = "-"
                    vals.append(f"{str(val):<{widths[i]}}")
                print("| " + " | ".join(vals) + " |")
            print("|" + "-" * (row_len-2) + "|")
            
        print("=" * row_len + "\n")

    def print_alpha_comparison_table(self):
        """
        Special table for alpha sensitivity.
        Group by budget, compare different alphas.
        """
        if not self.results:
            return
            
        # Define columns
        cols = ["alpha", "status", "high_ratio", "medium_ratio", "low_ratio", "time_ms"]
        headers = ["Alpha", "Status", "High %", "Med %", "Low %", "Time (ms)"]
        
        # Identify unique budgets - Use 'budget' field instead of 'budget_limit'
        budgets = sorted(list(set(r.get("budget") for r in self.results if r.get("budget") is not None)))
        
        # Calculate widths
        widths = [len(h) for h in headers]
        for r in self.results:
            for i, c in enumerate(cols):
                val = r.get(c, "")
                if isinstance(val, float):
                    if "ratio" in c: val = f"{val*100:.1f}%"
                    elif "time" in c: val = f"{val:.1f}"
                widths[i] = max(widths[i], len(str(val)))
                
        row_len = sum(widths) + len(cols)*3 + 1
        print("=" * row_len)
        print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |")
        print("|-" + "-|-".join("-" * w for w in widths) + "-|")
        
        for b in budgets:
            print(f"| Budget: {b:<{row_len - 12}} |")
            
            # Get rows for this budget
            rows = [r for r in self.results if r.get("budget") == b]
            # Sort by alpha descending (1.0 -> 0.3)
            rows.sort(key=lambda x: x.get('alpha', 0), reverse=True)
            
            for r in rows:
                vals = []
                for i, c in enumerate(cols):
                    val = r.get(c, "")
                    if isinstance(r.get(c), float):
                        if "ratio" in c: val = f"{r[c]*100:.1f}%"
                        elif "time" in c: val = f"{r[c]:.1f}"
                    vals.append(f"{str(val):<{widths[i]}}")
                print("| " + " | ".join(vals) + " |")
            print("|" + "-" * (row_len-2) + "|")
        print("=" * row_len + "\n")
