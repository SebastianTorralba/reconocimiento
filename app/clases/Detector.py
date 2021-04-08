import cv2
import time
import numpy as np
import os
import PIL
import pytesseract
import random
from datetime import datetime
from PIL import Image
from sklearn.cluster import KMeans


class Detector:

    def __init__(self):
        self.tammax = 1024
        pathActual = os.path.dirname(__file__)
        self.pathTemp = os.path.join(pathActual, 'tmp/')
        self.pathIdentificadas = os.path.join(pathActual, 'identificadas/')

    def identificar(self, pathImg):
        imgOriginal = Image.open(pathImg).convert('RGB')
        wpercent = (self.tammax / float(imgOriginal.size[0]))
        hsize = int((float(imgOriginal.size[1]) * float(wpercent)))
        imgNueva = imgOriginal.resize((self.tammax, hsize), PIL.Image.ANTIALIAS)
        imagen = cv2.cvtColor(np.array(imgNueva), cv2.COLOR_RGB2BGR)
        # imagen = imutils.resize(imagen, width=800)
        # cv2.imshow('Foto', imagen)
        imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Foto GRIS',imagenGris)
        imagenGris = cv2.bilateralFilter(imagenGris, 11, 17, 17)
        # cv2.imshow('Foto GRIS Filtro',imagenGris)
        edge = cv2.Canny(imagenGris, 170, 200)
        # cv2.imshow('CANNY', edge)
        cnts, nuevo = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        imagen1 = imagen.copy()
        cv2.drawContours(imagen1, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow("CANNY CONTORNOS", imagen1)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
        contLicencias = None
        imagen2 = imagen.copy()
        cv2.drawContours(imagen2, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow("Top 30 contornos", imagen2)
        count = 0
        nombre = 1
        licenciaImagen = None
        for i in cnts:
            perimetro = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimetro, True)
            if (len(approx) == 4):
                contLicencias = approx
                x, y, w, h = cv2.boundingRect(i)
                crp_img = imagen[y:y + h, x:x + w]
                tiempo = int(str(time.time()).replace('.', ''))
                self.pathTemp = self.pathTemp + str(nombre) + str(tiempo) + str(random.randint(0, 1024)) + '.png'
                esNegra = True
                try:
                    esNegra = self.esNegra(crp_img)
                except Exception as e:
                    respuesta = {"error": True, "patente": "", "errorDesc": repr(e)}
                    return respuesta
                if esNegra == True:
                    invertida = cv2.bitwise_not(crp_img)
                    cv2.imwrite(self.pathTemp, invertida)
                    nombre += 1
                else:
                    cv2.imwrite(self.pathTemp, crp_img)
                    nombre += 1
                break
        try:
            respuesta = []
            # cv2.drawContours(imagen, [contLicencias], -1, (0, 255, 0), 3)
            licenciaImagen = cv2.imread(self.pathTemp)
            #os.remove(self.pathTemp)
            # cv2.imshow("Encontrada", imagen)
            # cv2.imshow("Licencia", licenciaImagen)
            texto = pytesseract.image_to_string(licenciaImagen, lang='eng')
            if not (texto):
                respuesta = {"error": True, "patente": "", "errorDesc": "NODETECTO-ORC"}
                return respuesta
            else:
                tiempo = int(str(time.time()).replace('.', ''))
                self.pathIdentificadas = self.pathIdentificadas + str(tiempo) + str(random.randint(0, 1024)) + '.png'
                cv2.imwrite(self.pathIdentificadas, licenciaImagen)
                sinEspacios = texto.replace(" ", "")
                respuesta = {"error": False, "patente": sinEspacios, "errorDesc": ""}
                return respuesta
        except Exception as e:
            respuesta = {"error": True, "patente": "", "errorDesc": repr(e)}
            return respuesta

    def colorPredominante(self, cluster, centroids):
        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins=labels)
        hist = hist.astype("float")
        hist /= hist.sum()
        colores = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
        tam = len(colores)
        color = colores[tam - 1][1]
        col = (int(color[0]), int(color[1]), int(color[2]))
        return col

    def esNegra(self, imagen):
        reshape = imagen.reshape((imagen.shape[0] * imagen.shape[1], 3))
        cluster = KMeans(n_clusters=10).fit(reshape)
        colorPredo = self.colorPredominante(cluster, cluster.cluster_centers_)
        color = self.clasificar(colorPredo)
        return True if (color == 'negro') else False

    def clasificar(self, rgb_tuple):
        colores = {"blanco": (255, 255, 255), "negro": (0, 0, 0)}
        manhattan = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
        distancias = {k: manhattan(v, rgb_tuple) for k, v in colores.items()}
        color = min(distancias, key=distancias.get)
        return color
