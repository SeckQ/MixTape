import spotipy
import spotipy.util as util

import random



def autenticar_spotify(token):
	print('Conectando a Spotify*****************')
	sp = spotipy.Spotify(auth=token)
	#print(sp.current_user())
	return sp


def agregar_top_artistas(sp):
	print('Recuperando tus artistas favoritos*****************')
	nombre_top_artistas = []
	uri_top_artistas = []

	rangos = ['short_term', 'medium_term', 'long_term']
	for r in rangos:
		todos_datos_top_artistas = sp.current_user_top_artists(limit=50, time_range= r)
		datos_top_artistas = todos_datos_top_artistas['items']
		for datos_artista in datos_top_artistas:
			if datos_artista["name"] not in nombre_top_artistas:
				nombre_top_artistas.append(datos_artista['name'])
				uri_top_artistas.append(datos_artista['uri'])

	todos_datos_artistas_seguidos = sp.current_user_followed_artists(limit=50)
	datos_artistas_seguidos = (todos_datos_artistas_seguidos['artists'])
	for datos_artista in datos_artistas_seguidos["items"]:
		if datos_artista["name"] not in nombre_top_artistas:
			nombre_top_artistas.append(datos_artista['name'])
			uri_top_artistas.append(datos_artista['uri'])
	return uri_top_artistas



def agregar_top_canciones(sp, uri_top_artistas):
	print("Recuperando tus canciones favoritas*****************")
	uri_top_canciones = []
	for artista in uri_top_artistas:
		todos_datos_top_canciones = sp.artist_top_tracks(artista)
		datos_top_canciones = todos_datos_top_canciones['tracks']
		for datos_cancion in datos_top_canciones:
			uri_top_canciones.append(datos_cancion['uri'])
	return uri_top_canciones


def seleccionar_canciones(sp, uri_top_canciones, animo):
	
	print("Seleccionando las canciones *****************")
	uri_canciones_seleccionadas = []

	def agrupar(seq, tamano):
		return (seq[pos:pos + tamano] for pos in range(0, len(seq), tamano))

	random.shuffle(uri_top_canciones)
	for canciones in list(agrupar(uri_top_canciones, 50)):
		todos_datos_canciones = sp.audio_features(canciones)
		for datos_cancion in todos_datos_canciones:
			# try:
			# 	if animo < 0.10:
			# 		if (0 <= datos_cancion["valence"] <= (animo + 0.15)
			# 		and datos_cancion["danceability"] <= (animo * 8)
			# 		and datos_cancion["energy"] <= (animo * 10)):
			# 			uri_canciones_seleccionadas.append(datos_cancion["uri"])
			# 	elif 0.10 <= animo < 0.25:
			# 		if ((animo - 0.075) <= datos_cancion["valence"] <= (animo + 0.075)
			# 		and datos_cancion["danceability"] <= (animo * 4)
			# 		and datos_cancion["energy"] <= (animo * 5)):
			# 			uri_canciones_seleccionadas.append(datos_cancion["uri"])
			# 	elif 0.25 <= animo < 0.50:
			# 		if ((animo - 0.085) <= datos_cancion["valence"] <= (animo + 0.085)
			# 		and datos_cancion["danceability"] <= (animo * 3)
			# 		and datos_cancion["energy"] <= (animo * 3.5)):
			# 			uri_canciones_seleccionadas.append(datos_cancion["uri"])
			# 	elif 0.50 <= animo < 0.75:
			# 		if ((animo - 0.075) <= datos_cancion["valence"] <= (animo + 0.075)
			# 		and datos_cancion["danceability"] >= (animo / 2.5)
			# 		and datos_cancion["energy"] >= (animo / 2)):
			# 			uri_canciones_seleccionadas.append(datos_cancion["uri"])
			# 	elif 0.75 <= animo < 0.90:
			# 		if ((animo - 0.075) <= datos_cancion["valence"] <= (animo + 0.075)
			# 		and datos_cancion["danceability"] >= (animo / 2)
			# 		and datos_cancion["energy"] >= (animo / 1.75)):
			# 			uri_canciones_seleccionadas.append(datos_cancion["uri"])
			# 	elif animo >= 0.90:
			# 		if ((animo - 0.15) <= datos_cancion["valence"] <= 1
			# 		and datos_cancion["danceability"] >= (animo / 1.75)
			# 		and datos_cancion["energy"] >= (animo / 1.5)):
			# 			uri_canciones_seleccionadas.append(datos_cancion["uri"])
			try:
				if animo < 0.10:
					if (0 <= datos_cancion["valence"] <= (animo + 0.15)
							and datos_cancion["danceability"] <= (animo * 8)
							and datos_cancion["energy"] <= (animo * 10)):
						uri_canciones_seleccionadas.append(datos_cancion["uri"])

				# Enojado-->
				elif 0.10 <= animo < 0.40:
					if ((animo - 0.075) <= datos_cancion["valence"] <= (animo + 0.075)
							and datos_cancion["danceability"] <= (animo * 2)
							and datos_cancion["energy"] <= (animo * 3)):
						uri_canciones_seleccionadas.append(datos_cancion["uri"])

				# Neutral
				elif 0.40 <= animo < 0.60:
					if ((animo - 0.05) <= datos_cancion["valence"] <= (animo + 0.05)
							and datos_cancion["danceability"] >= (animo * 1.45)
							and datos_cancion["energy"] >= (animo * 1.45)):
						uri_canciones_seleccionadas.append(datos_cancion["uri"])
				# Feliz
				elif 0.60 <= animo <= 1:
					if ((animo - 0.15) <= datos_cancion["valence"] <= 1
							and datos_cancion["danceability"] >= (animo / 1.75)
							and datos_cancion["energy"] >= (animo / 1.5)):
						uri_canciones_seleccionadas.append(datos_cancion["uri"])

			except TypeError as te:
				continue
	return uri_canciones_seleccionadas


def crear_playlist(sp, uri_canciones_seleccionadas, animo):

	print("Creando la lista de reproduccion*****************")
	todos_datos_usuario = sp.current_user()
	id_usuario = todos_datos_usuario["id"]

	todos_datos_playlist = sp.user_playlist_create(id_usuario, "MixTape " + str(animo))
	print(todos_datos_playlist)
	id_playlist = todos_datos_playlist["id"]
	uri_playlist = todos_datos_playlist["uri"]
	url_playlist = todos_datos_playlist['external_urls']['spotify']
	nombre_playlist = todos_datos_playlist['name']
	print(nombre_playlist)
	#print(url_playlist)

	random.shuffle(uri_canciones_seleccionadas)
	sp.user_playlist_add_tracks(id_usuario, id_playlist, uri_canciones_seleccionadas[0:30])

	return uri_playlist, url_playlist, nombre_playlist, id_playlist
