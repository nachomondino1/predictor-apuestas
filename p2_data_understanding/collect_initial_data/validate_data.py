import pandas as pd

def validar_estructura(df_referencia, df_nuevo):
    # Comparar columnas
    columnas_faltantes = set(df_referencia.columns) - set(df_nuevo.columns)
    columnas_extra = set(df_nuevo.columns) - set(df_referencia.columns)

    errores = []
    if columnas_faltantes:
        errores.append(f"Faltan columnas: {columnas_faltantes}")
    if columnas_extra:
        errores.append(f"Columnas inesperadas: {columnas_extra}")
    
    # Comparar tipos de datos
    for columna in df_referencia.columns:
        if columna in df_nuevo.columns:
            tipo_esperado = df_referencia[columna].dtype
            tipo_actual = df_nuevo[columna].dtype
            if tipo_esperado != tipo_actual:
                errores.append(
                    f"Columna '{columna}': tipo esperado {tipo_esperado}, pero se encontró {tipo_actual}"
                )
    return errores

def validar_datos(df_referencia, df_nuevo):
    errores = []

    # Verificar valores NaN en columnas críticas
    for columna in df_referencia.columns:
        try:
            if df_nuevo[columna].isna().all():
                errores.append(f"La columna '{columna}' está completamente vacía.")
        except KeyError:
            pass
    
    # Validar valores dentro del rango esperado
    for columna in df_referencia.columns:
        if columna in df_nuevo.columns and pd.api.types.is_numeric_dtype(df_referencia[columna]):
            min_val = df_referencia[columna].min()
            max_val = df_referencia[columna].max()
            
            if (df_nuevo[columna] < min_val).any():
                errores.append(f"La columna '{columna}' tiene valores menores al mínimo esperado ({min_val}).")
            
            if (df_nuevo[columna] > max_val).any():
                errores.append(f"La columna '{columna}' tiene valores mayores al máximo esperado ({max_val}).")
    
    '''
    # Validar valores únicos o patrones específicos --> PODRIA SER RANGOS.
    for columna, valores_esperados in df_referencia.nunique().items():
        if columna in df_nuevo.columns:
            valores_unicos = df_nuevo[columna].nunique()
            if valores_unicos < valores_esperados:
                errores.append(
                    f"La columna '{columna}' tiene menos valores únicos ({valores_unicos}) que el ejemplo ({valores_esperados})."
                )
    '''

    return errores

def comparar_dataframes(df_referencia, df_nuevo):
    estructura_errores = validar_estructura(df_referencia, df_nuevo)
    datos_errores = validar_datos(df_referencia, df_nuevo)

    if not estructura_errores and not datos_errores:
        print("El nuevo DataFrame cumple con las expectativas.")
    else:
        print("Se encontraron los siguientes errores:")
        for error in estructura_errores + datos_errores:
            print(f" - {error}")

''' Para enviar alerta por Telegram o wpp.
import requests

def enviar_alerta(mensaje):
    token = "TOKEN_DE_TELEGRAM"
    chat_id = "CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": mensaje})

def enviar_alerta_si_hay_errores(errores):
    if errores:
        mensaje = "Se encontraron los siguientes errores:\n" + "\n".join(errores)
        enviar_alerta(mensaje)  # Reutiliza la función de alerta que vimos antes
'''

if __name__ == "__main__":

    '''
    # DataFrame de referencia
    df_referencia = pd.DataFrame({
        "campo_1": ["valor1", "valor2", "valor3"],
        "campo_2": [10, 20, 30],
        "campo_3": ["A", "B", "C"]
    })

    # Supongamos que este es el nuevo DataFrame extraído
    df_nuevo = pd.DataFrame({
        "campo_1": ["valor1", "valor2", None],
        "campo_2": [10, 20, 30],
        "campo_3": ["A", None, "C"]
    })
    '''

    df_referencia = pd.read_excel('data/england/p2_data_understanding/df_match.xlsx', index_col=0)
    df_nuevo = pd.read_excel('data/england/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx', index_col=0)

    # Comparar con el DataFrame de referencia
    comparar_dataframes(df_referencia, df_nuevo)
