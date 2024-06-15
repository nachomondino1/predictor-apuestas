import pandas as pd
from collections import defaultdict

def generate_cron_jobs(file_path):
    
    # Levanta schedules.xlsx
    df = pd.read_excel(file_path) # sheet_name='schedules'

    # Por fecha
    cron_jobs = []
    for _, row in df.iterrows():
        dt = row['date_mod']
        cron_expression = f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"
        cron_jobs.append((cron_expression, row['id_country']))

    return cron_jobs

def update_github_workflow(cron_jobs, workflow_path):
    workflow_content = """name: Run predictions

on:
  schedule:
"""
    for cron, _ in cron_jobs:
        workflow_content += f"    - cron: \"{cron}\"\n"

    workflow_content += "jobs:\n"

    for cron, country in cron_jobs:
        job_name = f"collect-data-job-{cron.replace(' ', '-').replace('*', 'star')}-{country}"

        workflow_content += f"""
  {job_name}:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          sparse-checkout: |
            github_actions/update_workflow/schedules.xlsx
            github_actions/update_workflow/update_workflow.py
            p6_deployment/collect_data.py
          sparse-checkout-cone-mode: false
          ref: prod  # Branch

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Install dependencies
        run: pip install pandas openpyxl

      - name: Run collect_data script
        run: python p6_deployment/collect_data.py --id_country {country}
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
