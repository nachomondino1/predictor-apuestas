import pandas as pd
import os
from collections import defaultdict

def generate_cron_jobs(file_path):
    
    # Levanta schedules.xlsx
    df = pd.read_excel(file_path) # sheet_name='schedules'

    # Formateo date y time como una sola columna y datetime --> si tengo separada date y time
    # df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time_cron'])

    # Por fecha
    cron_jobs = defaultdict(list)
    for _, row in df.iterrows():
        
        # La convierto en formato cron
        dt = row['date_mod']
        cron_expression = f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"
        cron_jobs[cron_expression].append(row['id_country'])

    return cron_jobs

def update_github_workflow(cron_jobs, workflow_path):
    workflow_content = """name: Run predictions

on:
"""
    for cron, countries in cron_jobs.items():
        l_countries = ','.join(map(str, countries))
        workflow_content += f"  schedule:\n    - cron: \"{cron}\"\n"
        workflow_content += f"""
jobs:
  collect-data-job-{cron.replace(" ", "-").replace("*", "star")}:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          sparse-checkout: |
            p6_deployment/data/schedules.xlsx
            p6_deployment/update_workflow.py
          sparse-checkout-cone-mode: false
          ref: prod  # Branch

      - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.x'

      - name: Install dependencies
      run: pip install pandas openpyxl
      
      - name: Run collect_data script
        run: python p6_deployment/collect_data.py --l_countries "{l_countries}"
"""

    with open(workflow_path, 'w') as file:
        file.write(workflow_content)

if __name__ == "__main__":
    # Defino argumentos
    file_path = 'github_actions/update_workflow/schedules.xlsx'
    workflow_path = '.github/workflows/main_next_matches.yml'

    # Formateo fecha como fecha de cronjob
    cron_jobs = generate_cron_jobs(file_path)

    # Actualizo workflow 'Collect Data' segun fechas
    update_github_workflow(cron_jobs, workflow_path)
