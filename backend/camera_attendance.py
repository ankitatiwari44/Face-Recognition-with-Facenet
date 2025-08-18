import csv
from datetime import datetime
import os

def mark_attendance(name, filename="Attendance.csv"):
    # If file does not exist, create it with headers
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])  # Correct headers
    
    # Get current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    # Read existing records to avoid duplicate for same person & date
    with open(filename, "r") as f:
        existing_data = f.readlines()

    names_today = [line.split(",")[0] for line in existing_data if dt_string in line]

    if name not in names_today:  # Prevent duplicate entry
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, dt_string, time_string])
