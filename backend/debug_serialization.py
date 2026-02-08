
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class StrategicBankAgent:
    bank_id: str
    name: str

def _clean_results(d: dict) -> dict:
    """Convert numpy arrays and other non-serializable types to JSON-safe."""
    out = {}
    for k, v in d.items():
        if k == "agents":
            continue
        if isinstance(v, np.ndarray):
            v_safe = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            out[k] = v_safe.tolist()
        elif isinstance(v, dict):
            out[k] = _clean_results(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v) if np.isfinite(v) else 0.0
        elif isinstance(v, list):
            cleaned_list = []
            for item in v:
                if isinstance(item, np.ndarray):
                    cleaned_list.append(
                        np.nan_to_num(item, nan=0.0, posinf=0.0, neginf=0.0).tolist()
                    )
                elif isinstance(item, (np.floating, float)):
                    cleaned_list.append(float(item) if np.isfinite(item) else 0.0)
                elif isinstance(item, (np.integer, int)):
                    cleaned_list.append(int(item))
                elif isinstance(item, dict):
                    cleaned_list.append(_clean_results(item))
                else:
                    cleaned_list.append(item)
            out[k] = cleaned_list
        else:
            out[k] = v
    return out

def test():
    agents = [StrategicBankAgent("1", "Bank 1")]
    timeline = {
        "steps": [1, 2],
        "decisions": [["ROLL_OVER"], ["WITHDRAW"]],
        "values": [np.float64(1.0), float('inf')]
    }
    results = {
        "agents": agents,
        "timeline": timeline,
        "scalar": np.int64(10)
    }

    print("Cleaning...")
    try:
        cleaned = _clean_results(results)
        print("Cleaned keys:", cleaned.keys())
        print("Dumping JSON...")
        json_str = json.dumps(cleaned)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
