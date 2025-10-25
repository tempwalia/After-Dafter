import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Generate synthetic call center data for demo purposes."""
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Number of records
    n_records = 1000
    
    # Generate synthetic data
    data = {
        'customer_id': range(1, n_records + 1),
        'call_duration': np.random.normal(180, 60, n_records),  # mean 3 mins
        'previous_contacts': np.random.poisson(2, n_records),
        'time_of_day': np.random.choice(['morning', 'afternoon', 'evening'], n_records),
        'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], n_records),
        'customer_segment': np.random.choice(['A', 'B', 'C'], n_records, p=[0.2, 0.5, 0.3]),
        'last_contact_days': np.random.exponential(30, n_records),
        'rpc_label': np.random.binomial(1, 0.3, n_records)  # 30% success rate
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic features
    df['call_duration'] = df['call_duration'].clip(min=10)  # minimum 10 seconds
    df['last_contact_days'] = df['last_contact_days'].clip(min=1).round()
    
    # Save to CSV
    output_path = data_dir / "synthetic_callcenter_accounts.csv"
    df.to_csv(output_path, index=False)
    print(f"Sample data created at: {output_path}")
    print(f"Records generated: {len(df)}")
    
    return output_path

if __name__ == "__main__":
    create_sample_data()