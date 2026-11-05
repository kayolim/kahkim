"""Game parameters from the 2025 'K-Bottle' rules (IOG 문제 설명서). Update if the round changes."""

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
JOB_SIZE = 1000          # 1 job = 1,000 units
MAX_JOBS = 500           # per product per day

# Production (per product line, per day)
SETUP_COST = {"P1": 5_000_000, "P2": 2_000_000}
REGULAR_HOURS = 70       # 7,000 makespan units
WAGE = 100_000           # per hour (=100 makespan units), floor(makespan/100)
OT_WAGE = 200_000
TIME_FILE = {"P1": "t_500_20_{day}.csv", "P2": "t2_500_20_{day}.csv"}

# Finished goods
PRICE = {"P1": 250, "P2": 350}
FG_HOLD = {"P1": 30, "P2": 30}        # per unit per day
STOCKOUT = {"P1": 250, "P2": 250}     # lost-sale penalty per unit (plus lost revenue)

# Bill of materials: units of material per unit of product
BOM = {"P1": {"M1": 1, "M2": 2}, "P2": {"M2": 2, "M3": 1}}

# Materials: initial stock, order cost, unit cost, lead time (days), holding (per unit-day)
MATERIALS = {
    "M1": {"init": 300_000, "order_cost": 1_000_000, "types": {"NORMAL": (30, 3)}, "hold": 1},
    "M2": {"init": 3_000_000, "order_cost": 3_000_000,
           "types": {"NORMAL": (25, 8), "URGENT": (50, 2)}, "hold": 1},
    "M3": {"init": 500_000, "order_cost": 2_000_000, "types": {"NORMAL": (50, 5)}, "hold": 2},
}


def labor_cost(makespan: int) -> int:
    h = int(makespan) // 100
    return min(h, REGULAR_HOURS) * WAGE + max(h - REGULAR_HOURS, 0) * OT_WAGE


def production_cost(product: str, makespan: int, jobs: int) -> int:
    return 0 if jobs == 0 else SETUP_COST[product] + labor_cost(makespan)
