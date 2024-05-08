from fastapi import FastAPI
import requests
import pandas as pd
import ast
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

########## 1. OBTENCIÓN DE LOS DATASETS #############
'''URL a los datos en Github '''
horas_jugadas_parquet = 'https://github.com/rcarmonae/HENRY_PI01_RCARMONA_MAYO2024/raw/main/DATASETS/F01_PlaytimeGenre.parquet'
horas_jugadas_usuario_parquet = 'https://github.com/rcarmonae/HENRY_PI01_RCARMONA_MAYO2024/raw/main/DATASETS/F02_UserForGenre.parquet'
game_reviews_parquet = 'https://github.com/rcarmonae/HENRY_PI01_RCARMONA_MAYO2024/raw/main/DATASETS/F03_UsersRecommend.parquet'


'''Solicita el contenido de las URL'''
res_horas_jugadas = requests.get(horas_jugadas_parquet)
res_horas_jugadas_usuario = requests.get(horas_jugadas_usuario_parquet)
res_game_reviews = requests.get(game_reviews_parquet)


'''Convierte los csv a dataframe'''
horas_jugadas = pd.read_parquet(BytesIO(res_horas_jugadas.content))
horas_jugadas_usuario = pd.read_parquet(BytesIO(res_horas_jugadas_usuario.content))
game_reviews = pd.read_parquet(BytesIO(res_game_reviews.content))
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
def UsersRecommend(year : int):
    '''Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)'''
    
    # Filtrar el DataFrame por año, recomendación positiva y sentimientos positivos/neutrales
    filtered_df = game_reviews[
        (game_reviews['release_year'] == year) & 
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

@app.get('/recomendacion_juego/{item_id}')
def recomendacion_juego(item_id : int):
    # Supongamos que 'df' es el DataFrame global que contiene la información de los juegos
    df = game_reviews
    df['item_id'] = df['item_id'].astype(int)

    #global df  # Utiliza el DataFrame global

    # Formatea la columna de géneros para que se lean como listas
    if isinstance(df.loc[0, 'genres'], str):
        df['genres'] = df['genres'].apply(eval)

    try:
        game_info = df[df['item_id'] == item_id].iloc[0]
    except IndexError:
        return "No se encontró el juego con el item_id proporcionado."

    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genres'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index)

    features = pd.concat([df[['sentiment_analysis', 'recommend', 'release_year']], genre_df], axis=1)
    features.dropna(inplace=True)

    # Eliminar duplicados para asegurar que cada juego es único
    features = features[~df['item_id'].duplicated(keep='first')]

    # Almacenar los índices antes de eliminar el juego dado para posterior referencia
    valid_indices = features.index

    target_features = features.loc[df['item_id'] == item_id]
    features = features.drop(index=target_features.index)

    # Calcular la similitud del coseno entre el juego dado y todos los otros juegos
    cosine_sim = cosine_similarity(features, target_features)

    # Asignar los puntajes de similitud al DataFrame que solo contiene los índices válidos
    valid_df = df.loc[valid_indices].drop(index=target_features.index)
    valid_df['similarity_score'] = cosine_sim[:, 0]

    # Asegurar que las recomendaciones sean juegos únicos
    valid_df = valid_df.groupby('item_id').max('similarity_score')

    return valid_df.nlargest(5, 'similarity_score')
