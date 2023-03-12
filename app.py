import cv2
import imutils
import keras
import numpy as np
import spotipy.util as util
from flask import Flask, request, render_template, session
from keras.preprocessing.image import image_utils

from mixtape_funciones import autenticar_spotify, agregar_top_artistas, agregar_top_canciones, seleccionar_canciones, \
    crear_playlist

id_cliente = "0de9f853905c4196b5bc0ad0a0ffb927"
clave_cliente = "a0ad730545104ba0996633fe68f21b36"
uri_redireccion = "http://localhost:8010"

alcance = 'user-library-read user-top-read playlist-modify-public user-follow-read'

username = "seck2401"
token = util.prompt_for_user_token(username, alcance, id_cliente, clave_cliente, uri_redireccion)

app = Flask(__name__)
app.secret_key = 'clavesecreta'

preguntaGlobal = 1

@app.route('/', methods=['GET', 'POST'])
def index():  # put application's code here
    return render_template('index.html')




# @app.route('/camera', methods=['GET', 'POST'])
# def camera():
#     i = 0
#
#     GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
#     model = tf.keras.models.load_model('final_model.h5')
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface2.xml')
#     output = []
#     cap = cv2.VideoCapture(0)
#     while (i <= 90):
#         ret, img = cap.read()
#         faces = face_cascade.detectMultiScale(img, 1.05, 5)
#
#         for x, y, w, h in faces:
#             face_img = img[y:y + h, x:x + w]
#
#             resized = cv2.resize(face_img, (224, 224))
#             reshaped = resized.reshape(1, 224, 224, 3) / 255
#             predictions = model.predict(reshaped)
#
#             max_index = np.argmax(predictions[0])
#
#             emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
#             predicted_emotion = emotions[max_index]
#             output.append(predicted_emotion)
#
#             cv2.rectangle(img, (x, y), (x + w, y + h), GR_dict[1], 2)
#             cv2.rectangle(img, (x, y - 40), (x + w, y), GR_dict[1], -1)
#             cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         i = i + 1
#
#         cv2.imshow('LIVE', img)
#         key = cv2.waitKey(1)
#         if key == 27:
#             cap.release()
#             cv2.destroyAllWindows()
#             break
#     print(output)
#     cap.release()
#     cv2.destroyAllWindows()
#     final_output1 = st.mode(output)
#     return render_template("botones.html", final_output=final_output1)


def predecir_emocion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detecciones = faceNet.forward()

    caras = []
    locs = []
    preds = []
    for i in range(0, detecciones.shape[2]):

        # Fija un umbral para determinar que la detección es confiable
        # Tomando la probabilidad asociada en la deteccion

        if detecciones[0, 0, i, 2] > 0.4:
            # Toma el bounding box de la detección escalado
            # de acuerdo a las dimensiones de la imagen
            box = detecciones[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Valida las dimensiones del bounding box
            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0

            # Se extrae el rostro y se convierte BGR a GRAY
            # Finalmente se escala a 224x244
            cara = frame[Yi:Yf, Xi:Xf]
            cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
            cara = cv2.resize(cara, (48, 48))
            cara2 = image_utils.img_to_array(cara)
            cara2 = np.expand_dims(cara2, axis=0)

            # Se agrega los rostros y las localizaciones a las listas
            caras.append(cara2)
            locs.append((Xi, Yi, Xf, Yf))

            pred = emotionModel.predict(cara2)
            preds.append(pred[0])

    return (locs, preds)


clases = ['enojado', 'feliz', 'neutral', 'triste']
prototxtDirectorio = r"resources\deploy.prototxt"
pesosDirectorio = r"resources\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtDirectorio, pesosDirectorio)

modeloEmociones = keras.models.load_model('modeloCara100.h5')


@app.route('/camara', methods=['GET', 'POST'])
def camara():
    output = []
    i = 0
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    carita = []
    etiquetas_list = []
    confianzas_list = []
    i = 0
    etiquetas = {}  # Diccionario vacío para guardar etiquetas y sus valores de confianza

    while (i <= 90):
    #while True:
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=640)
        (locs, preds) = predecir_emocion(frame, faceNet, modeloEmociones)
        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            (enojado, feliz, neutral, triste) = pred

            label = ''
            # Se agrega la probabilidad en el label de la imagen
            label = "{}: {:.0f}%".format(clases[np.argmax(pred)],
                                         max(enojado, feliz, neutral, triste) * 100)

            cv2.rectangle(frame, (Xi, Yi - 40), (Xf, Yi), (255, 0, 0), -1)
            cv2.putText(frame, label, (Xi + 5, Yi - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

            etiqueta = clases[np.argmax(pred)]
            confianza = max(enojado, feliz, neutral, triste) * 100
            etiquetas_list.append(etiqueta)
            confianzas_list.append(confianza)

            # Actualizar el diccionario con la etiqueta y su valor de confianza
            if label.split(':')[0] in etiquetas:
                etiquetas[label.split(':')[0]]['count'] += 1
                etiquetas[label.split(':')[0]]['confidence'] += max(enojado, feliz, neutral, triste)
            else:
                etiquetas[label.split(':')[0]] = {'count': 1,'confidence': max(enojado, feliz, neutral, triste)}

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i = i + 1

            print(label)
    print(carita)

    # Imprimir resultados
    for etiqueta, valores in etiquetas.items():
        print(etiqueta, '->', valores['count'], '->', valores['confidence'])

    # Contar etiquetas y acumular confianzas
    conteo_etiquetas = {}
    suma_confianzas = {}

    for etiqueta, confianza in zip(etiquetas_list, confianzas_list):
        if etiqueta not in conteo_etiquetas:
            conteo_etiquetas[etiqueta] = 0
            suma_confianzas[etiqueta] = 0
        conteo_etiquetas[etiqueta] += 1
        suma_confianzas[etiqueta] += confianza

    # Identificar etiqueta más frecuente y calcular confianza promedio
    etiqueta_mas_frecuente = max(conteo_etiquetas, key=conteo_etiquetas.get)
    confianza_promedio = suma_confianzas[etiqueta_mas_frecuente] / conteo_etiquetas[etiqueta_mas_frecuente]
    print("-------------------------------------------------------------------")
    print("Etiqueta más frecuente:", etiqueta_mas_frecuente)
    print("Confianza promedio:", confianza_promedio)

    #preguntaGlobal = int(session.get('pregunta'))
    preguntaGlobal = 1
    print('************************PG', preguntaGlobal)
    # print("***************", session.get('pregunta'))
    if preguntaGlobal == 1:
        if etiqueta_mas_frecuente == "triste":
            confianza_promedio = (confianza_promedio / 100) * 0.1
        elif etiqueta_mas_frecuente == "enojado":
            confianza_promedio = (confianza_promedio / 100) * 0.3 + 0.10
        elif etiqueta_mas_frecuente == "neutral":
            confianza_promedio = (confianza_promedio / 100) * 0.2 + 0.40
        elif etiqueta_mas_frecuente == "feliz":
            confianza_promedio = (confianza_promedio / 100) * 0.4 + 0.60

    else:
        if etiqueta_mas_frecuente == "triste":
            confianza_promedio = (confianza_promedio / 100) * 0.39 + 0.61
        elif etiqueta_mas_frecuente == "neutral":
            confianza_promedio = (confianza_promedio / 100) * 0.19 + 0.41
        elif etiqueta_mas_frecuente == "feliz":
            confianza_promedio = ((confianza_promedio / 100) * 0.4)

    print("-------------------------------------------------------------------")
    print("Etiqueta más frecuente:", etiqueta_mas_frecuente)
    print("Confianza promedio:", confianza_promedio)

    session['ponderacionFinal'] = confianza_promedio

    return render_template("botones.html", salida_final=etiqueta_mas_frecuente, ponderacion = confianza_promedio)


@app.route('/templates/musica', methods=['GET'])
def musica():
    ponderacion_animo = session.get('ponderacionFinal')
    return render_template("musica.html", ponderacion_animo = ponderacion_animo)


@app.route('/templates/musica', methods=['POST'])
def spotify_playlist():
    animo = request.form['mi-campo']
    animo = float(animo)

    spotipy_autenticacion = autenticar_spotify(token)
    top_artistas = agregar_top_artistas(spotipy_autenticacion)
    top_canciones = agregar_top_canciones(spotipy_autenticacion, top_artistas)
    canciones_seleccionadas = seleccionar_canciones(spotipy_autenticacion, top_canciones, animo)
    playlist1 = crear_playlist(spotipy_autenticacion, canciones_seleccionadas, animo)
    uri_playlist,url_playlist, nombre_playlist, id_playlist = playlist1
    session['uri_playlistFinal'] = uri_playlist
    session['nombre_playlistFinal'] = nombre_playlist
    session['url_playlistFinal'] = url_playlist
    session['id_playlistFinal'] = id_playlist
    #print("*************************",uri_playlist)
    #print("*************************",url_playlist)
    return render_template('playlist.html', lista_uri=uri_playlist,lista_url=url_playlist, nombre_playlist = nombre_playlist , id_playlist = id_playlist)


@app.route('/templates/botones', methods=['GET', 'POST'])
def buttons():
    return render_template("botones.html")


@app.route('/templates/equipo', methods=['GET', 'POST'])
def join():
    return render_template("equipo.html")


@app.route('/templates/funciones', methods=['GET', 'POST'])
def features():
    return render_template("funciones.html")


@app.route('/templates/playlist', methods=['GET', 'POST'])
def playlist():
    return render_template("playlist.html")

@app.route('/templates/reproduccionweb', methods=['GET', 'POST'])
def reproduccionWeb():
    nombre_playlist_reproduccion_web = session.get('nombre_playlistFinal')
    url_playlist_reproduccion_web = session.get('url_playlistFinal')
    id_playlist_reproduccion_web = session.get('id_playlistFinal')
    return render_template("reproduccionWeb.html", nombre_playlist_reproduccion_web = nombre_playlist_reproduccion_web, url_playlist_reproduccion_web = url_playlist_reproduccion_web, id_playlist_reproduccion_web = id_playlist_reproduccion_web)

@app.route('/templates/reproduccionlocal', methods=['GET', 'POST'])
def reproduccionLocal():
    uri_playlist_reproduccion_local = session.get('uri_playlistFinal')
    nombre_playlist_reproduccion_local = session.get('nombre_playlistFinal')
    url_playlist_reproduccion_local = session.get('url_playlistFinal')
    id_playlist_reproduccion_local = session.get('id_playlistFinal')
    return render_template("reproduccionLocal.html", nombre_playlist_reproduccion_local = nombre_playlist_reproduccion_local, url_playlist_reproduccion_local = url_playlist_reproduccion_local, id_playlist_reproduccion_local = id_playlist_reproduccion_local, uri_playlist_reproduccion_local = uri_playlist_reproduccion_local)

@app.route('/templates/retroalimentacion', methods=['GET'])
def retroalimentacion():
    id_playlist_reproduccion_retro = session.get('id_playlistFinal')
    return render_template("retroalimentación.html", id_playlist_reproduccion_retro = id_playlist_reproduccion_retro)

@app.route('/templates/retroalimentacion', methods=['POST'])
def registrar_pregunta():
    pregunta = request.form.get('selected_option')
    #print('*****************', pregunta)
    session['pregunta'] = pregunta
    #print("***************", session.get('pregunta'))
    return render_template('retroalimentación.html')

if __name__ == '__main__':
    app.run()
