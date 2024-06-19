import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from datetime import datetime

def create_schedule(df_next_matches, min_before_match=15):
    """
    Obtengo fecha y hora y pais para el cual correr collect_predictions.py
    """
    hours_utc_argentina = 3  # Argentina es UTC-3
    min_delay_schedule = 0  # La action tarda hasta 5 minutos desde schedule time en correrse

    # Obtener fechas unicas y paises 
    df = df_next_matches.loc[:, ['date', 'id_country']]  # Selecciono algunas columnas 
    df = df.drop_duplicates()   # Obtener los registros únicos

    # Restar x minutos a cada hora
    delta_time = min_before_match + min_delay_schedule - hours_utc_argentina * 60  # [minutos]
    delta = pd.to_timedelta(delta_time, unit='m')
    df['date_mod'] = df['date'] - delta    # Restar el timedelta a cada valor de la columna 'Hora'

    # Exportar schedules.xlsx
    df.to_excel('automatize_predict/update_predictions/schedules.xlsx', index=False)
    return df

def generate_cron_jobs(df):
    """Convierte cada fecha en una expresión cron"""
    # Por fecha
    cron_jobs = []
    for _, row in df.iterrows():
        dt = row['date_mod']
        cron_expression = f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"
        cron_jobs.append((cron_expression, row['id_country']))

    return cron_jobs

def create_cronjob_action(cron_jobs, workflow_path):
    
    # Schedule
    workflow_content = """name: Update predictions with line-ups

on:
  schedule:
"""
    for cron, _ in cron_jobs:
        workflow_content += f"    - cron: \"{cron}\"\n"

    # Permissions
    workflow_content += "\npermissions: write-all\n"

    # Jobs
    workflow_content += "\njobs:\n"

    # Por job
    for cron, id_country in cron_jobs:
        job_name = f"collect-data-job-{cron.replace(' ', '-').replace('*', 'star')}-{id_country}"

        workflow_content += f"""
  {job_name}:
    runs-on: ubuntu-latest

    # Asocio cron con job
    if: github.event.schedule == '{cron}'

    steps:

      # Clono repo de producto en maquina ubuntu donde corre el workflow
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          ref: prod  # Branch

      # Instalo dependencias
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'
 
      - name: Set up cache for pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements_mnm.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          pip install -r requirements_mnm.txt

      # Ejecución de collect_predictions.py
      - name: Run collect_data script
        run: |
          python automatize_predict/collect_predictions.py 0.05 "[{id_country}]" '{{"run_missing": false, "data_unders": true, "data_prep": true, "modeling": true, "export": true}}'
        
      # Push to Github
      - name: Commit and push all changes
        env:
          # Set the environment variable GITHUB_TOKEN if needed for push authentication
          GITHUB_TOKEN: ${{ secrets.TOKEN }}
        run: |
          git config --global user.name 'github-actions[bot]'
          git config --global user.email 'github-actions[bot]@users.noreply.github.com'
          git add .  # Agregar todos los cambios
          git commit -m "Update predictions with line-ups"
          git push origin prod

      # Dispatch
      - name: Dispatch event to second repository
        env:
          GITHUB_TOKEN: ${{ secrets.TOKEN }}
        run: |
          python automatize_predict/dispatch_event/dispatch_event.py
"""

    with open(workflow_path, 'w') as file:
        file.write(workflow_content)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino argumentos
    # minutos_a_restar = 15 #int(sys.argv[1]) # Definir la cantidad de minutos a restar
    df_next_matches = pd.read_excel('p6_deployment/data/historial_predicciones.xlsx') # Levantar proximos partidos --> tengo que garantizar que sean los partidos de los proximos 15 dias.
    df_next_matches = df_next_matches[df_next_matches['date'].dt.date >= datetime.now().date()]
    workflow_path = '.github/workflows/update_predictions.yml'

    # Creo schedules.xlsx
    df_schedules = create_schedule(df_next_matches) # pd.read_excel(file_path) # sheet_name='schedules' # file_path = 'github_actions/automatize_predict/schedules.xlsx'

    # Formateo fecha como fecha de cronjob
    cron_jobs = generate_cron_jobs(df_schedules)

    # Actualizo workflow 'Collect Data' segun fechas
    create_cronjob_action(cron_jobs, workflow_path)
