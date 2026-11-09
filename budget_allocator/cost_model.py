from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class SecurityLevelCost:
    """
    Cost definition for a specific security level.
    """
    time_cost: float  # e.g., milliseconds per block or relative unit
    key_cost: float   # e.g., key usage units or bytes per block

class CostModel:
    """
    Manages cost definitions for different security levels.
    """
    
    def __init__(self, costs: Optional[Dict[str, SecurityLevelCost]] = None):
        """
        Args:
            costs: Dictionary mapping level names to costs.
                   If None, loads default values.
        """
        if costs is None:
            self.costs = self._get_default_costs()
        else:
            self.costs = costs

    @staticmethod
    def _get_default_costs() -> Dict[str, SecurityLevelCost]:
        """
        Official cost model matching q2_base.yaml.
        High (L1): 10 units
        Medium (L2): 3 units
        Low (L3): 1 unit
        None (L0): 0 units
        """
        return {
            "high": SecurityLevelCost(time_cost=10.0, key_cost=10.0),
            "medium": SecurityLevelCost(time_cost=3.0, key_cost=5.0),
            "low": SecurityLevelCost(time_cost=1.0, key_cost=0.0),
            "none": SecurityLevelCost(time_cost=0.0, key_cost=0.0)
        }

    def get_cost(self, level: str, budget_type: str = "time") -> float:
        """
        Get the cost for a single block at a specific level.
        """
        if level == "none" and level not in self.costs:
            return 0.0
        if level not in self.costs:
            raise ValueError(f"Unknown security level: {level}")
            
        cost_entry = self.costs[level]
        
        if budget_type == "time":
            return cost_entry.time_cost
        elif budget_type == "key":
            return cost_entry.key_cost
        else:
            raise ValueError(f"Unknown budget type: {budget_type}")
