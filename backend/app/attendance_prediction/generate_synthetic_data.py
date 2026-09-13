"""
generate_synthetic_data.py

Purpose:
--------
Generates a SYNTHETIC (artificially created) prototype dataset for
the Event Attendance Prediction module.

IMPORTANT:
This data is NOT real. It is created using simple, reasonable rules
plus random noise, only so we can build and test our ML pipeline
before real historical event data is available.

Output file: synthetic_event_data_v1.csv
"""

import random
import csv

# Fixed seed so the same dataset can be reproduced every time
random.seed(42)

NUM_ROWS = 200  # size of our prototype dataset

event_categories = ["Workshop", "Concert", "Hackathon", "Sports", "Tech Talk", "Cultural Fest"]

rows = []

for event_id in range(1, NUM_ROWS + 1):
    category = random.choice(event_categories)

    # Ticket price depends loosely on category (concerts/fests cost more)
    if category in ["Concert", "Cultural Fest"]:
        ticket_price = random.choice([0, 100, 200, 300, 500])
    else:
        ticket_price = random.choice([0, 0, 50, 100])  # many free workshops/talks

    # Venue capacity varies by event type
    if category == "Sports":
        venue_capacity = random.randint(500, 2000)
    elif category in ["Concert", "Cultural Fest"]:
        venue_capacity = random.randint(300, 1500)
    else:
        venue_capacity = random.randint(50, 300)

    # Promotion days: how long before the event it was advertised
    promotion_days = random.randint(3, 30)

    # Registrations tend to increase with more promotion days,
    # but are capped near venue capacity, plus some randomness
    base_registrations = (promotion_days * random.uniform(8, 15))
    num_registrations = int(min(base_registrations, venue_capacity * random.uniform(0.7, 1.0)))
    num_registrations = max(num_registrations, 10)  # avoid unrealistic near-zero values

    # Actual attendance: usually a fraction of registrations (not everyone shows up),
    # with random noise, and never exceeding venue capacity
    show_up_rate = random.uniform(0.6, 0.95)  # 60%-95% of registered people actually attend
    noise = random.uniform(-0.05, 0.05)       # small random variation
    actual_attendance = int(num_registrations * (show_up_rate + noise))

    # Safety limits: attendance can't be negative or exceed venue capacity
    actual_attendance = max(actual_attendance, 0)
    actual_attendance = min(actual_attendance, venue_capacity)

    rows.append([
        event_id,
        category,
        ticket_price,
        venue_capacity,
        promotion_days,
        num_registrations,
        actual_attendance
    ])

# Write to CSV
output_path = "synthetic_event_data_v1.csv"
header = ["event_id", "event_category", "ticket_price", "venue_capacity",
          "promotion_days", "num_registrations", "actual_attendance"]

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Synthetic dataset created: {output_path}")
print(f"Total rows generated: {len(rows)}")
print("NOTE: This is SYNTHETIC data for prototyping only. Not real event records.")
