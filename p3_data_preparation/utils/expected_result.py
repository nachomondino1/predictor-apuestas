import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p3_data_preparation import construct_data
from p3_data_preparation.select_data import determine_country_competitions
from p4_modeling import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score
import pickle
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier

var_resp = 'result'
d_countries = {
    48: ["england", '2025-04-20'],
    55: ["france", '2025-04-21'], 
    59: ["germany", '2025-04-21'],
    77: ["italy", '2025-04-20'],
    148: ["spain", '2025-04-21']
    }

data_unders = False
l_models = [LogisticRegression(), SVC(), RandomForestClassifier(),  XGBClassifier(), GradientBoostingClassifier()]  # MLPClassifier()

if data_unders:
    df = pd.DataFrame()

    # DATA UNDERS
    for id_country, lista in d_countries.items():
        country = lista[0]
        iteration_date = lista[1]

        # Levanto df_integrated del pais.
        df_country = pd.read_excel(f'data/{country}/p3_data_preparation/{iteration_date}/df_integrated.xlsx', index_col=0)
        print(df_country.shape)

        df = pd.concat([df, df_country], axis=0)
        print(df.shape)

    df.to_excel('data/df_integrated_all.xlsx')
else:
    df = pd.read_excel('data/all/p3_data_preparation/2025-04-22/df_integrated.xlsx', index_col=0)
    print(df)

# DATA PREPARATION
# Treat nan: Elimino nan values en varibles predictoras
df.dropna(subset=['expected_goals_(xg)_home', 'expected_goals_(xg)_away'], inplace=True)

# Construct data
df = construct_data.determine_result(df)
df['dif_goals'] = df['expected_goals_(xg)_home'] - df['expected_goals_(xg)_away']
df['sum_goals'] = df['expected_goals_(xg)_home'] + df['expected_goals_(xg)_away']

# Select data
# Filtro por competencias --> lo mejor es sin filtar...
# d_comps = determine_country_competitions(id_country=-1)
# comp_to_selec = d_comps['comp_solo_liga']
# df = df[df['id_competition'].isin(comp_to_selec)]

# Dejo solamente expected_goals_home y expected_goals_away como variables predictoras 
cols = ['id_country', 'id_competition', 'expected_goals_(xg)_home', 'expected_goals_(xg)_away', 'dif_goals', 'sum_goals', 'result']
df = df.loc[:, cols] #  # Capaz puedo dejar id_country e id_competition tmb?

# MODELING
# Dividir en sets: train, val, test?
X, y = df.drop(var_resp, axis=1), df[var_resp]
print(X.shape, y.shape)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# Entreno modelo de ml
for modelo in l_models:
    print(f"Modelo: {modelo}")
    model, params, d_metrics, results = build_model.select_best_hiperparameters(
                    modelo, 
                    X_train=X_train, 
                    y_train=y_train, 
                    X_val=X_val, 
                    y_val=y_val, 
                    k=10, 
                    refit='cross_entropy_loss', # probé f1_score, f1_score_draw y cross_entropy
                    bayes=False, 
                    verbose=1
                    )

    # Predigo en X_test
    y_pred = model.predict(X_test)

    # Evaluo precision (y el resultado que predice seria mi "expected_result" capturando mejor la info de x_goals)
    d_metrics = {
        # 'error': -log_loss(y_test, y_pred_prob, labels=[0, 1, 2]),
        'test_accuracy': accuracy_score(y_test, y_pred) * 100,
        'recall': recall_score(y_test, y_pred, average='macro') * 100,
        'f1_score': f1_score(y_test, y_pred, average='macro') * 100,
    }
    print("Metricas finales:", d_metrics)


# Exporto modelo (para usar en construccion durante el train)
pickle.dump(model, open("./data/expected_result_2.pkl", "wb"))
