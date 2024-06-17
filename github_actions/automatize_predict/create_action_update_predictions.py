import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from collections import defaultdict

def create_schedule(df_next_matches, minutos_a_restar=15):
    """
    Obtengo fecha y hora y pais para el cual correr collect_predictions.py
    """
    # Obtener fechas unicas y paises 
    df = df_next_matches.loc[:, ['date', 'id_country']]  # Selecciono algunas columnas 
    df = df.drop_duplicates()   # Obtener los registros únicos

    # Restar x minutos a cada hora
    delta = pd.to_timedelta(minutos_a_restar, unit='m')
    df['date_mod'] = df['date'] - delta    # Restar el timedelta a cada valor de la columna 'Hora'

    # Exportar schedules.xlsx
    df.to_excel('github_actions/automatize_predict/schedules.xlsx', index=False)
    return df

def generate_cron_jobs(df):
    # Por fecha
    cron_jobs = []
    for _, row in df.iterrows():
        dt = row['date_mod']
        cron_expression = f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"
        cron_jobs.append((cron_expression, row['id_country']))

    return cron_jobs

def create_cronjob_action(cron_jobs, workflow_path):
    workflow_content = """name: Update predictions with line-ups

on:
  schedule:
"""
    for cron, _ in cron_jobs:
        workflow_content += f"    - cron: \"{cron}\"\n"

    workflow_content += "jobs:\n"

    for cron, id_country in cron_jobs:
        job_name = f"collect-data-job-{cron.replace(' ', '-').replace('*', 'star')}-{id_country}"

        workflow_content += f"""
  {job_name}:
    runs-on: ubuntu-latest

    permissions:                # Job-level permissions configuration starts here
      contents: write           # 'write' access to repository contents

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          sparse-checkout: |
            github_actions/automatize_predict
          sparse-checkout-cone-mode: false
          ref: prod  # Branch

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Install dependencies
        run: pip install pandas openpyxl

      - name: Run collect_data script
        run: python github_actions/automatize_predict/collect_predictions.py 0.05 [{id_country}]

      - name: Commit and push predictions.xlsx
        run: |
          git config --global user.name 'github-actions[bot]'
          git config --global user.email 'github-actions[bot]@users.noreply.github.com'
          git add p6_deployment/data/predicciones.xlsx
          git commit -m "Add updated predicciones.xlsx"
          git push
        env:
          # Set the environment variable GITHUB_TOKEN if needed for push authentication
          GITHUB_TOKEN: ${{ secrets.TOKEN }}
"""

    with open(workflow_path, 'w') as file:
        file.write(workflow_content)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino argumentos
    # minutos_a_restar = 15 #int(sys.argv[1]) # Definir la cantidad de minutos a restar
    df_next_matches = pd.read_excel('p6_deployment/data/predicciones.xlsx') # Levantar proximos partidos    
    workflow_path = '.github/workflows/update_predictions.yml'

    # Creo schedules.xlsx
    df_schedules = create_schedule(df_next_matches) # pd.read_excel(file_path) # sheet_name='schedules' # file_path = 'github_actions/automatize_predict/schedules.xlsx'

    # Formateo fecha como fecha de cronjob
    cron_jobs = generate_cron_jobs(df_schedules)

    # Actualizo workflow 'Collect Data' segun fechas
    create_cronjob_action(cron_jobs, workflow_path)
