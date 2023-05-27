import pandas as pd
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier

def feature_selection(df):  # Es sin variables categoricas

    X, y = df.drop('equipo_ganador', axis=1), df['equipo_ganador']

    # Ajusta un modelo de bosque aleatorio a los datos
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Evalúa la importancia de las características
    importances = rf.feature_importances_

    # Selecciona las características con importancia por encima de un umbral
    threshold = 0.05
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_new = selector.transform(X)

    # Muestra la importancia de las características seleccionadas
    support = selector.get_support()
    for feature, importance, supported in zip(X.columns, importances, support):
        if supported:
            print(f"{feature}: {importance:.3f} (selected)")
        else:
            print(f"{feature}: {importance:.3f}")

    # Ajusta un nuevo modelo de bosque aleatorio solo con las características seleccionadas
    rf_new = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_new.fit(X_new, y)


def main():
    df = pd.read_excel('/Users/nachomondino/Desktop/df_constructed.xlsx', index_col=0)

    df_correlation_matrix = df.drop('equipo_ganador', axis=1).corr()  # OJO que no tiene en cuenta las variables categoricas... y si quiero tenerlas en cuenta como "equipo ganador"
    df_correlation_matrix.to_excel('/Users/nachomondino/Desktop/correlation_matrix.xlsx')

    df.drop(['dif_pases', 'dif_pases_comp', 'dif_remates_a_puerta', 'dif_tarjetas_amarillas', 'dif_ataques_pelig'], inplace=True, axis=1)


    # feature_selection(df)

# main()
