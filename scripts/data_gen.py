#!/usr/bin/env python3
"""
Generate realistic synthetic call center data.

Outputs a CSV with per-call records including timestamps, customer/agent info,
queue/channel, wait/handle times, holds/transfers, outcomes, satisfaction,
and a binary rpc_label suitable for classification demos.

Usage (examples):
    python scripts/data_gen.py --rows 25000 --start 2025-01-01 --end 2025-06-30 \
            --out data/synthetic_callcenter_accounts.csv --seed 42 --rpc-rate 0.025

Notes:
- Distributions are parameterized to look realistic, with correlations
  (e.g., long wait => higher abandon & lower satisfaction).
- Requires: Python 3.11+, numpy, pandas.
"""
from __future__ import annotations
import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic call center data")
    p.add_argument("--rows", type=int, default=20000, help="Number of call records to generate")
    p.add_argument("--start", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default="2025-10-01", help="End date (YYYY-MM-DD, exclusive)")
    # Default to the existing filename used by notebooks/app so it plugs in seamlessly
    p.add_argument("--out", type=str, default="data/synthetic_callcenter_accounts.csv", help="Output CSV path")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--rpc-rate", type=float, default=0.025, help="Target average RPC per call (e.g., 0.025 => ~2.5%)")
    return p.parse_args()


def weighted_choice(rng: np.random.Generator, items: list, weights: list) -> any:
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    idx = rng.choice(len(items), p=probs)
    return items[idx]


def sample_call_times(rng: np.random.Generator, n: int, start_dt: datetime, end_dt: datetime) -> np.ndarray:
    """Sample start datetimes with realistic intraday volume patterns.
    Heavier 9am-6pm, lighter nights. Weekends lower volume.
    """
    seconds_range = int((end_dt - start_dt).total_seconds())
    # Draw candidate seconds and adjust acceptance by hour-of-day + weekday
    times = []
    while len(times) < n:
        sec = rng.integers(0, seconds_range)
        dt = start_dt + timedelta(seconds=int(sec))
        hour = dt.hour
        dow = dt.weekday()  # 0=Mon
        # Hour weight: peak business hours, shoulder mornings/evenings, low nights
        hour_w = {
            **{h: 0.3 for h in range(0, 7)},   # night low
            **{h: 0.7 for h in range(7, 9)},   # morning ramp
            **{h: 1.4 for h in range(9, 12)},  # morning peak
            **{h: 1.2 for h in range(12, 14)}, # lunch dip
            **{h: 1.5 for h in range(14, 18)}, # afternoon peak
            **{h: 0.8 for h in range(18, 21)}, # evening taper
            **{h: 0.4 for h in range(21, 24)}, # late
        }[hour]
        # Weekday weight: Mon-Thu normal, Fri slightly lower, weekend lowest
        dow_w = [1.1, 1.05, 1.0, 1.0, 0.9, 0.6, 0.5][dow]
        accept_prob = hour_w * dow_w / 1.5  # scale to <= ~1
        if rng.random() < min(accept_prob, 1.0):
            times.append(dt)
    return np.array(times, dtype='datetime64[ns]')


def generate(rows: int, start: str, end: str, seed: int, rpc_rate: float = 0.025) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    # Entity pools
    n_customers = max(5000, rows // 3)
    n_agents = 60
    customers = np.arange(100000, 100000 + n_customers)
    agents = np.arange(500, 500 + n_agents)

    teams = ["North", "South", "East", "West"]
    skills = ["Billing", "Tech", "Sales", "Retention"]
    queues = ["Billing", "Technical Support", "Sales", "Retention"]
    channels = ["phone", "chat", "email"]

    # Agent attributes
    agent_team = rng.choice(teams, size=n_agents, p=[0.27, 0.26, 0.24, 0.23])
    agent_skill = rng.choice(skills, size=n_agents, p=[0.30, 0.35, 0.20, 0.15])

    # Calls start times
    call_start = sample_call_times(rng, rows, start_dt, end_dt)

    # Assign customers/agents
    customer_id = rng.choice(customers, size=rows)
    agent_id = rng.choice(agents, size=rows)
    agent_team_map = {a: t for a, t in zip(agents, agent_team)}
    agent_skill_map = {a: s for a, s in zip(agents, agent_skill)}

    # Channels & queues
    channel = rng.choice(channels, size=rows, p=[0.75, 0.2, 0.05])
    queue = rng.choice(queues, size=rows, p=[0.35, 0.4, 0.15, 0.10])

    # Pre-dial, per-customer latent & historical features (introduce real signal)
    cust_contactability = rng.beta(2.0, 8.0, size=n_customers)  # skewed toward lower contactability
    cust_age_days = rng.integers(60, 2000, size=n_customers)
    cust_prior_calls_30d = rng.poisson(lam=1.0 + 3.0 * cust_contactability, size=n_customers)
    # Map per-call
    contactability_map = {cid: c for cid, c in zip(customers, cust_contactability)}
    age_map = {cid: a for cid, a in zip(customers, cust_age_days)}
    prior_calls_map = {cid: v for cid, v in zip(customers, cust_prior_calls_30d)}
    contactability_score = np.array([contactability_map[c] for c in customer_id])
    account_age_days = np.array([age_map[c] for c in customer_id])
    prior_calls_30d = np.array([prior_calls_map[c] for c in customer_id])

    # Wait time (sec) depends on queue and time of day (busier => longer waits)
    # Extract hour of day safely from numpy datetime64[ns] array
    hours = pd.DatetimeIndex(call_start).hour.values
    base_wait = rng.gamma(shape=2.0, scale=15.0, size=rows)  # typical around 30 sec
    hour_factor = np.where((hours >= 9) & (hours <= 17), 1.3, 0.8)
    queue_factor = np.select(
        [queue == 'Technical Support', queue == 'Billing', queue == 'Retention', queue == 'Sales'],
        [1.4, 1.2, 1.3, 0.9],
        default=1.0,
    )
    wait_time_sec = (base_wait * hour_factor * queue_factor).clip(0, 1200).astype(int)

    # Handle duration (sec) - lognormal-ish
    handle_time_sec = rng.lognormal(mean=4.6, sigma=0.5, size=rows)  # median ~100 sec
    # Longer for tech/retention
    handle_time_sec *= np.select(
        [queue == 'Technical Support', queue == 'Retention'],
        [1.25, 1.15],
        default=1.0,
    )
    handle_time_sec = handle_time_sec.clip(30, 3600).astype(int)

    # Hold time and transfers
    hold_time_sec = (rng.exponential(scale=10, size=rows) * (wait_time_sec > 60)).astype(int)
    transfers = rng.poisson(lam=np.where(queue == 'Technical Support', 0.25, 0.1), size=rows)

    # Abandon probability increases with wait
    abandon_prob = np.clip(0.02 + (wait_time_sec / 180.0)**1.2 * 0.08, 0, 0.65)
    abandoned = rng.random(rows) < abandon_prob

    # Outcomes
    outcomes = np.empty(rows, dtype=object)
    outcomes[abandoned] = 'abandoned'
    not_abd = ~abandoned
    # resolved vs escalated vs callback
    res_prob = np.clip(0.75 - 0.00015 * np.maximum(wait_time_sec - 60, 0) - 0.0001 * transfers, 0.35, 0.9)
    esc_prob = np.clip(0.10 + 0.0002 * np.maximum(wait_time_sec - 45, 0), 0.05, 0.40)
    cb_prob = 1.0 - res_prob - esc_prob
    choice = rng.random(rows)
    outcomes[not_abd & (choice < res_prob)] = 'resolved'
    outcomes[not_abd & (choice >= res_prob) & (choice < res_prob + esc_prob)] = 'escalated'
    outcomes[not_abd & (choice >= res_prob + esc_prob)] = 'callback'

    # RPC label (Right Party Contact) — rare event (~2-3% per call) with modifiers.
    # We construct a base probability per call then scale it to match the target rpc_rate.
    ch_factor = np.select([channel == 'phone', channel == 'chat', channel == 'email'], [1.0, 0.6, 0.2], default=1.0)
    q_factor_rpc = np.select(
        [queue == 'Sales', queue == 'Retention', queue == 'Billing', queue == 'Technical Support'],
        [1.2, 0.9, 0.8, 0.7],
        default=0.8,
    )
    hour_factor_rpc = np.where((hours >= 9) & (hours <= 19), 1.1, 0.9)
    # Longer waits reduce chance of RPC slightly
    wait_factor_rpc = 1.0 / (1.0 + (wait_time_sec / 600.0))
    # Customer pre-dial signal: higher contactability and some recency/volume signal increase RPC chance
    cust_signal = 0.6 + 0.8 * contactability_score + 0.1 * np.tanh((prior_calls_30d - 1.0) / 3.0)
    base_raw = ch_factor * q_factor_rpc * hour_factor_rpc * wait_factor_rpc * cust_signal
    # Zero probability if abandoned
    not_abd_mask = (~abandoned).astype(float)
    mean_raw = (base_raw * not_abd_mask).mean()
    scale = (rpc_rate / mean_raw) if mean_raw > 0 else 0.0
    rpc_prob = np.clip(base_raw * scale, 0.0, 0.3) * not_abd_mask
    rpc_label = (rng.random(rows) < rpc_prob).astype(int)

    # Customer segments & satisfaction
    segment = rng.choice(['A', 'B', 'C'], size=rows, p=[0.25, 0.5, 0.25])
    # Satisfaction 1-5 decreased by wait time and abandonment; improved if resolved
    sat_base = rng.normal(loc=3.8, scale=0.7, size=rows)
    sat_penalty = 0.002 * wait_time_sec + 0.5 * (outcomes == 'abandoned')
    sat_boost = 0.3 * (outcomes == 'resolved')
    satisfaction = np.clip(np.round(sat_base - sat_penalty + sat_boost), 1, 5).astype(int)

    # Costs and revenue proxies
    cost_per_min = np.select(
        [queue == 'Technical Support', queue == 'Retention', queue == 'Billing', queue == 'Sales'],
        [1.8, 1.6, 1.2, 1.0],
        default=1.2,
    )
    handle_minutes = handle_time_sec / 60.0
    cost_usd = np.round(cost_per_min * handle_minutes + 0.05 * transfers + 0.02 * (hold_time_sec/60.0), 2)
    revenue_usd = np.where(queue == 'Sales', np.round(rng.exponential(scale=20, size=rows), 2), 0.0)

    # End time
    call_end = (pd.to_datetime(call_start) + pd.to_timedelta(wait_time_sec + handle_time_sec + hold_time_sec, unit='s')).values

    df = pd.DataFrame({
        'call_id': np.arange(1, rows + 1),
        'customer_id': customer_id,
        'agent_id': agent_id,
        'agent_team': [agent_team_map[a] for a in agent_id],
        'agent_skill': [agent_skill_map[a] for a in agent_id],
        'channel': channel,
        'queue': queue,
        'call_start': pd.to_datetime(call_start),
        'call_end': pd.to_datetime(call_end),
        'wait_time_sec': wait_time_sec,
        'handle_time_sec': handle_time_sec,
        'hold_time_sec': hold_time_sec,
        'transfers': transfers,
        'outcome': outcomes,
        'satisfaction': satisfaction,
        'segment': segment,
        'rpc_label': rpc_label,
        'cost_usd': cost_usd,
        'revenue_usd': revenue_usd,
        # Pre-dial features added
        'contactability_score': np.round(contactability_score, 4),
        'account_age_days': account_age_days,
        'prior_calls_30d': prior_calls_30d,
    })

    # Derivatives
    df['duration_sec'] = df['wait_time_sec'] + df['handle_time_sec'] + df['hold_time_sec']
    df['date'] = df['call_start'].dt.date
    df['day_of_week'] = df['call_start'].dt.day_name()
    df['hour'] = df['call_start'].dt.hour
    df['week'] = df['call_start'].dt.isocalendar().week.astype(int)

    # Sort by start time for readability
    df = df.sort_values('call_start').reset_index(drop=True)
    return df


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(args.rows, args.start, args.end, args.seed, rpc_rate=args.rpc_rate)
    df.to_csv(out_path, index=False)

    # Quick summary printed
    print(f"Generated: {len(df):,} rows -> {out_path}")
    print("Date range:", df['call_start'].min(), "to", df['call_start'].max())
    print("Queues:")
    print(df['queue'].value_counts().to_string())
    print("Outcomes:")
    print(df['outcome'].value_counts().to_string())
    print("RPC rate:", round(df['rpc_label'].mean()*100, 2), "%")


if __name__ == "__main__":
    main()
