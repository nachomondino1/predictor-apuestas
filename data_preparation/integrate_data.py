# Juntar entidades jugadores, partido y atrib_jugadores
import pandas as pd
import re

def main():  # Tengo que eliminar acentos en el nombre de los jugadores en df_jug....
    # Levanto dataset
    df_part = pd.read_excel("/Users/nachomondino/Desktop/df_formated.xlsx", index_col=0)
    df_jug = pd.read_excel("/Users/nachomondino/Desktop/df_jug_formated.xlsx", index_col=0)
    print(df_part.info())
    print(df_jug.info())

    l_var = ['l_jug_tit_loc'] #, 'l_jug_tit_vis'] #, 'l_jug_ausentes_loc', 'l_jug_ausentes_vis']

    # Por partido
    for i in range(len(df_part))[-10:]:

        # Obtengo año del partido  # e.g. 2023
        year = df_part.loc[i, 'fecha'].year
        print(list(df_part.loc[i]))
        print(f'Año del partido: {year}')

        # Por variable (l_jug_tit_loc, l_jug_tit_vis, l_jug_ausentes_loc, l_jug_ausentes_vis)
        for var in l_var:

            # Reinicio variables
            l_prom_edad, l_prom_alt, l_prom_rating = [], [], []

            try:
                # convierto str a lista
                l_jug = eval(df_part.loc[i, var])  # e.g. ["Dibu", ..., "Messi"]

                # Por jugador
                for jug in l_jug:

                    # Filtro df_jug por fecha, nombre del jugador, equipo? (OJO aqui, puede haber jugadores con mismo nombre...)
                    l_nombre_a_buscar = jug.split()  # e.g. "Rodriguez D. (G) (C)"
                    # equipo_a_buscar = f"{l_nombre_a_buscar[1]} {l_nombre_a_buscar[0]}"

                    nombre_a_buscar = f"{l_nombre_a_buscar[1]} {l_nombre_a_buscar[0]}"
                    print(f' Nombre: {jug.split()} --> Nombre a buscar: {nombre_a_buscar}')

                    # Busco overall_rating, edad, altura, valor de mercado
                    print(df_jug[(df_jug['nombre'] == nombre_a_buscar) & (df_jug['fecha'] == year)])

                    try:
                        idx = df_jug[(df_jug['nombre'] == nombre_a_buscar) & (df_jug['fecha'] == year)].index[0]
                        print(list(df_jug.loc[idx]))

                        l_prom_edad.append(df_jug.loc[idx, 'edad'])
                        l_prom_alt.append(df_jug.loc[idx, 'altura'])
                        l_prom_rating.append(df_jug.loc[idx, 'overall_rating'])
                    except:
                        pass
            except TypeError:
                print(f"No hay lista de jugadores en {df_part.loc[i, var]}")
                pass

            try:
                # Promedio edad, overall_rating y valor de mercado y creo nuevas columnas en df_part
                df_part.loc[i, 'prom_edad'] = sum(l_prom_edad) / len(l_prom_edad)
                df_part.loc[i, 'prom_alt'] = sum(l_prom_alt) / len(l_prom_alt)
                df_part.loc[i, 'prom_rating'] = sum(l_prom_rating) / len(l_prom_rating)
            except:
                pass

    df_part.to_excel('/Users/nachomondino/Desktop/prueba.xlsx')
    return df_part

main()