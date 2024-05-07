from fastapi import FastAPI
import requests
import pandas as pd
import ast


########## 1. OBTENCIÓN DE LOS DATASETS #############
'''URL a los datos en Github '''
horas_jugadas_parquet = 'https://github.com/rcarmonae/HENRY_PI01_RCARMONA_MAYO2024/blob/f44880a6abff8d46fd5a624f6a1be2c6fcbb2b73/DATASETS/F01_PlaytimeGenre.parquet'
horas_jugadas_usuario_parquet = 'https://github.com/rcarmonae/HENRY_PI01_RCARMONA_MAYO2024/blob/main/DATASETS/F02_UserForGenre.parquet'
game_reviews_parquet = 'https://github.com/rcarmonae/HENRY_PI01_RCARMONA_MAYO2024/blob/main/DATASETS/F03_UsersRecommend.parquet'


'''Solicita el contenido de las URL'''
res_horas_jugadas = requests.get(horas_jugadas_parquet)
res_horas_jugadas_usuario = requests.get(horas_jugadas_usuario_parquet)
res_game_reviews = requests.get(game_reviews_parquet)


'''Convierte los csv a dataframe'''
horas_jugadas = pd.read_parquet(res_horas_jugadas)
horas_jugadas_usuario = pd.read_parquet(res_horas_jugadas_usuario)
game_reviews = pd.read_parquet(res_game_reviews)

########## 2. DESARROLLO API: DISPONIBILIZAR LOS DATOS USANDO FastAPI #############
app = FastAPI()
@app.get('/')
def bienvenida():
    return 'HENRY: Proyecto Individual I - Rosa Carmona'

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str):
    '''Devuelve el año con más horas jugadas para dicho género'''
    
    # Filtrar por un género específico
    horas_jugadas_filtrado = horas_jugadas[horas_jugadas['genres'] == genero]
    
    # Agrupar por años y sumar las horas jugadas de cada uno
    total_playtime_year = horas_jugadas_filtrado.groupby('release_year')['playtime_forever'].sum()

    # Encontrar el año con el mayor tiempo jugado para el género dado
    year_most_played = total_playtime_year.idxmax()
       
    return {f"Año de lanzamiento con más horas jugadas para el género {genero}: {int(year_most_played)}"}

@app.get('/UserForGenre/{genero}')
def UserForGenre(genero :str):
    '''Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.'''
    
    # Filtrar por un género específico
    horas_jugadas_usuario_filtrado = horas_jugadas_usuario[horas_jugadas_usuario['genres'] == genero]

    # Agrupar por años y sumar las horas jugadas de cada uno
    total_playtime_user = horas_jugadas_usuario_filtrado.groupby(['user_id','release_year'])['playtime_forever'].sum()

    # Agrupar otra vez por usuario
    user_group =total_playtime_user.groupby('user_id').sum()

    # Encontrar el jugador con el mayor tiempo jugado para el género dado
    user_most_played = user_group.idxmax()

    # Segmenta el DF anterior con el resumen de usuario y año, para obtener solo los registros del usuario que más horas jugó 
    user_years = total_playtime_user.loc[user_most_played]

    # Prepara el dataframe para imprimirlo en el formato adecuado
    user_years = user_years.reset_index() # Genera un nuevo índice para liberar el año del registro
    user_years['release_year'] = user_years['release_year'].astype(int) # Convierte los años de float a int
    user_years['playtime_forever'] = user_years['playtime_forever'].astype(int) # Convierte las horas jugadas de float a int
    user_years = user_years[user_years['playtime_forever'] != 0] # Remueve las filas de los años en los que las horas jugadas = 0
    list_of_dicts = user_years.to_dict('records') # Convierte el DataFrame a una lista de diccionarios
    formatted_entries = [f"{{Año: {d['release_year']}, Horas: {d['playtime_forever']}}}" for d in list_of_dicts] # Crear una lista de strings formateados como diccionarios
    formatted_string = ", ".join(formatted_entries) # Unir la lista en un string único

    # Formatear el mensaje para imprimir
    #final_message = f"Usuario con más horas jugadas para el género {genero}: {user_most_played}, Horas jugadas: [{formatted_string}]"
    return {f"Usuario con más horas jugadas para el género {genero}: {user_most_played}, Horas jugadas: [{formatted_string}]"}

@app.get('/UserRecommend/{año}')
def UsersRecommend(año : int):
    '''Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)'''
    
    # Filtrar el DataFrame por año, recomendación positiva y sentimientos positivos/neutrales
    filtered_df = game_reviews[
        (game_reviews['release_year'] == año) & 
        (game_reviews['recommend'] == True) & 
        (game_reviews['sentiment_analysis'] >= 1)
    ]
    
    # Contar las recomendaciones por juego
    recommendation_counts = filtered_df['item_id'].value_counts().head(3)
    
    # Crear un DataFrame para los resultados
    top_games = pd.DataFrame({
        'item_id': recommendation_counts.index,
        'recommendations': recommendation_counts.values      
     })
    
    # join entre top games y filtered_df para asignar los nombres (titulos) de los items
    top_game_titles = pd.merge(top_games[['item_id','recommendations']], filtered_df[['item_id','title']], left_on='item_id', right_on='item_id', how='left')
    top_game_titles = top_game_titles.drop_duplicates(subset=['item_id'])
    top_game_titles.reset_index(drop=True, inplace=True)
    final_message = f"[Puesto 1: {top_game_titles.loc[0,'title']}, Puesto 2: {top_game_titles.loc[1,'title']}, Puesto 3: {top_game_titles.loc[2,'title']}]"
   
    return final_message
