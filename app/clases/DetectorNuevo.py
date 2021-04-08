import cv2
import sys
import os
import time
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from clases.util.LocalUtils import detect_lp
from sklearn.cluster import KMeans
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from tensorflow import Session, ConfigProto
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DetectorNuevo:

    def resetPaths(self):
        self.pathTemp = 'tmp/'
        self.pathIdentificadas = 'identificadas/'
        self.pathNoSegmentadas = 'noseg/'

    def iniModel(self, _config):
        self.sesion = Session(config=_config)
        self.graph = self.sesion.graph
        set_session(self.sesion)
        json_patentes = open('rs_deeplearn/modelos/patentes.json', 'r')
        patentes_json = json_patentes.read()
        json_patentes.close()
        self.modelo = model_from_json(patentes_json)
        self.modelo.load_weights("rs_deeplearn/modelos/weights/patentes.h5")
        print("[INFO] Modelo Patentes Cargado...")
        json_caracteres = open('rs_deeplearn/modelos/caracteres-nuevos.json', 'r')
        caracteres_json = json_caracteres.read()
        json_caracteres.close()
        self.modeloCaracteres = model_from_json(caracteres_json)
        self.modeloCaracteres.load_weights("rs_deeplearn/modelos/weights/caracteres-nuevos.h5")
        print("[INFO] Modelo Caracteres Cargado...")
        self.labels = LabelEncoder()
        self.labels.classes_ = np.load('rs_deeplearn/clases/caracteres-nuevos.npy')
        print("[INFO] Labels Caracteres Cargados...")

    def __init__(self):
        self.resetPaths()
        config = self.keras_resource()
        self.iniModel(config)

    def keras_resource(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        return config

    def predecirDesdeModelo(self, imagen, modelo, etiquetas):
        imagen = cv2.resize(imagen, (80, 80))
        imagen = np.stack((imagen,) * 3, axis=-1)
        with self.sesion.graph.as_default():
            with self.sesion.as_default():
                prediccion = etiquetas.inverse_transform([np.argmax(modelo.predict(imagen[np.newaxis, :]))])
                return prediccion

    def prepocesarImagen(self, imagen_path, redim=False):
        img = cv2.imread(imagen_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if redim:
            img = cv2.resize(img, (224, 224))
        return img

    def getPatente(self, imagen_path, Dmax=600, Dmin=200):
        vehiculo = self.prepocesarImagen(imagen_path)
        ratio = float(max(vehiculo.shape[:2])) / min(vehiculo.shape[:2])
        lado = int(ratio * Dmin)
        unida_dim = min(lado, Dmax)
        with self.sesion.graph.as_default():
            with self.sesion.as_default():
                _, LpImg, _, cor = detect_lp(self.modelo, vehiculo, unida_dim, lp_threshold=0.5)
                return LpImg, cor

    def sortContornos(self, cnts, inverso=False):
        i = 0
        delimCuadros = [cv2.boundingRect(c) for c in cnts]
        (cnts, delimCuadros) = zip(*sorted(zip(cnts, delimCuadros), key=lambda b: b[1][i], reverse=inverso))
        return cnts

    def autoCanny(self, imagen, sigma=0.33):
        v = np.median(imagen)
        menor = int(max(0, (1.0 - sigma) * v))
        mayor = int(min(255, (1.0 + sigma) * v))
        afilada = cv2.Canny(imagen, menor, mayor)
        return afilada

    def patenteNegra(self, patente, patenteOriginal, extension, caracteres):
        patente2 = cv2.bitwise_not(patente)
        resultado = {}
        resultado["caracteres"] = 0
        resultado["patente"] = ""
        resultado["errorDesc"] = "NO SEGMENTADO"
        resultado["error"] = True
        cont, _ = cv2.findContours(patente, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Pruebaroi = patente2.copy()
        digitoW, digitoH = 30, 60
        for c in self.sortContornos(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            proporcion = h / w
            if 1 <= proporcion <= 3.5:
                if h / patente.shape[0] >= 0.5:
                    cv2.rectangle(Pruebaroi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    actualNum = patente2[y:y + h, x:x + w]
                    actualNum = cv2.resize(actualNum, dsize=(digitoW, digitoH))
                    _, curr_num = cv2.threshold(actualNum, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    caracteres.append(actualNum)
        resultado["caracteres"] = len(caracteres)
        if (len(caracteres) > 0):
            # figura = plt.figure(figsize=(15, 3))
            # columnas = len(caracteres)
            # grilla = gridspec.GridSpec(ncols=columnas, nrows=1, figure=figura)
            armado = ''
            for i, caracter in enumerate(caracteres):
                # figura.add_subplot(grilla[i])
                titulo = np.array2string(self.predecirDesdeModelo(caracter, self.modeloCaracteres, self.labels))
                # plt.title('{}'.format(titulo.strip("'[]"), fontsize=20))
                armado += titulo.strip("'[]")
            if (armado):
                tiempo = int(str(time.time()).replace('.', ''))
                self.pathIdentificadas = self.pathIdentificadas + str(tiempo) + str(random.randint(0, 1024))
                cv2.imwrite(self.pathIdentificadas + extension, patenteOriginal)
                cv2.imwrite(self.pathIdentificadas + '_proc' + extension, patente2)
            resultado["patente"] = armado.replace(" ", "")
            resultado["errorDesc"] = ""
            resultado["error"] = False
        else:
            tiempo = int(str(time.time()).replace('.', ''))
            self.pathNoSegmentadas = self.pathNoSegmentadas + str(tiempo) + str(random.randint(0, 1024))
            cv2.imwrite(self.pathNoSegmentadas + extension, patenteOriginal)
            cv2.imwrite(self.pathNoSegmentadas + '_proc' + extension, patente2)
        return resultado

    def patenteBlanca(self, patente, patenteOriginal, extension, caracteres):
        resultado = {}
        resultado["caracteres"] = 0
        resultado["patente"] = ""
        resultado["errorDesc"] = "NO SEGMENTADO"
        resultado["error"] = True
        cont, _ = cv2.findContours(patente, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pruebaRoi = patente.copy()
        digitoW, digitoH = 30, 60
        for c in self.sortContornos(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            proporcion = h / w
            if 1 <= proporcion <= 3.5:
                if h / patente.shape[0] >= 0.5:
                    cv2.rectangle(pruebaRoi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    actualNum = patente[y:y + h, x:x + w]
                    actualNum = cv2.resize(actualNum, dsize=(digitoW, digitoH))
                    _, curr_num = cv2.threshold(actualNum, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    caracteres.append(actualNum)
        resultado["caracteres"] = len(caracteres)
        if (len(caracteres) > 0):
            # figura = plt.figure(figsize=(15, 3))
            # columnas = len(caracteres)
            # grilla = gridspec.GridSpec(ncols=columnas, nrows=1, figure=figura)
            armado = ''
            for i, caracter in enumerate(caracteres):
                # figura.add_subplot(grilla[i])
                titulo = np.array2string(self.predecirDesdeModelo(caracter, self.modeloCaracteres, self.labels))
                # plt.title('{}'.format(titulo.strip("'[]"), fontsize=20))
                armado += titulo.strip("'[]")
            if (armado):
                tiempo = int(str(time.time()).replace('.', ''))
                self.pathIdentificadas = self.pathIdentificadas + str(tiempo) + str(random.randint(0, 1024))
                cv2.imwrite(self.pathIdentificadas + extension, patenteOriginal)
                cv2.imwrite(self.pathIdentificadas + '_proc' + extension, patente)
            resultado["patente"] = armado.replace(" ", "")
            resultado["errorDesc"] = ""
            resultado["error"] = False
        else:
            tiempo = int(str(time.time()).replace('.', ''))
            self.pathNoSegmentadas = self.pathNoSegmentadas + str(tiempo) + str(random.randint(0, 1024))
            cv2.imwrite(self.pathNoSegmentadas + extension, patenteOriginal)
            cv2.imwrite(self.pathNoSegmentadas + '_proc' + extension, patente)
        return resultado

    def colorPredominante(self, cluster, centroids):
        etiquetas = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins=etiquetas)
        hist = hist.astype("float")
        hist /= hist.sum()
        colores = sorted([(porcentaje, color) for (porcentaje, color) in zip(hist, centroids)])
        tam = len(colores)
        color = colores[tam - 1][1]
        col = (int(color[0]), int(color[1]), int(color[2]))
        return col

    def clasificar(self, rgb_tuple):
        colores = {"blanco": (255, 255, 255), "negro": (0, 0, 0)}
        manhattan = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
        distancias = {k: manhattan(v, rgb_tuple) for k, v in colores.items()}
        color = min(distancias, key=distancias.get)
        return color

    def esNegra(self, imagen):
        reshape = imagen.reshape((imagen.shape[0] * imagen.shape[1], 3))
        cluster = KMeans(n_clusters=10).fit(reshape)
        colorPredo = self.colorPredominante(cluster, cluster.cluster_centers_)
        color = self.clasificar(colorPredo)
        return True if (color == 'negro') else False

    def identificar(self, pathImg):
        self.resetPaths()
        resultado = {}
        try:
            LpImg, cor = self.getPatente(pathImg)
        except Exception as e:
            resultado = {"caracteres": 0, "error": True, "patente": "", "errorDesc": repr(e)}
            return resultado
        if (len(LpImg)):
            licenciaImagen = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
            gris = cv2.cvtColor(licenciaImagen, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gris, (7, 7), 0)
            binario = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            licenciaFinal = cv2.morphologyEx(binario, cv2.MORPH_DILATE, kernel3)
            extension = pathlib.Path(pathImg).suffix
            if self.esNegra(licenciaImagen):
                resultado = self.patenteNegra(licenciaFinal, licenciaImagen, extension, caracteres=[])
            else:
                resultado = self.patenteBlanca(licenciaFinal, licenciaImagen, extension, caracteres=[])
        return resultado
