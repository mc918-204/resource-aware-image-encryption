from typing import List, Dict, Any

class TaskMetrics:
    """
    Metrics specific to the resource-aware task.
    """
    
    @staticmethod
    def calculate_level_distribution(package_payload: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate percentage of blocks at each security level.
        """
        total = len(package_payload)
        if total == 0:
            return {}
            
        counts = {"high": 0, "medium": 0, "low": 0}
        for item in package_payload:
            lvl = item.get('level', 'low')
            counts[lvl] = counts.get(lvl, 0) + 1
            
        return {k: v / total * 100 for k, v in counts.items()}

    @staticmethod
    def calculate_budget_usage(actual_cost: float, total_budget: float) -> float:
        """
        Calculate budget usage percentage.
        """
        if total_budget <= 0:
            return 0.0
        return (actual_cost / total_budget) * 100.0
