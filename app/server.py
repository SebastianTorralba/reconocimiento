#!/usr/bin/python
import os
import time
from datetime import datetime
from flask_restful import reqparse, Api, Resource
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from clases.DetectorFinal import DetectorFinal

pathActual = os.path.dirname(__file__)
detectorNuevo = DetectorFinal()
CARPETA_SUBIDA = os.path.join(pathActual, 'uploads/')
EXTENSIONES = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = CARPETA_SUBIDA
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('username', type=str)
parser.add_argument('password', type=str)


def archivoPermitido(archivoNombre):
    return '.' in archivoNombre and \
           archivoNombre.rsplit('.', 1)[1].lower() in EXTENSIONES


class getPatente(Resource):
    def post(self):
        args = parser.parse_args()
        un = str(args['username'])
        pw = str(args['password'])
        return jsonify(u=un, p=pw)


@app.route('/getPatente', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'imagen' not in request.files:
            response = make_response("no hay archivo", 200)
            response.mimetype = "text/plain"
            return response
        archivo = request.files['imagen']
        if archivo.filename == '':
            response = make_response("enviar una imagen", 200)
            response.mimetype = "text/plain"
            return response
        if archivo and archivoPermitido(archivo.filename):
            tiempo = int(str(time.time()).replace('.', ''))
            archivoNombre = secure_filename(str(tiempo) + archivo.filename)
            fecha = datetime.now()
            fechaPath = fecha.strftime('%d-%m-%Y')
            pathUpload = os.path.join(app.config['UPLOAD_FOLDER'], fechaPath)
            pathUploadCompress = os.path.join(app.config['UPLOAD_FOLDER'], fechaPath)
            pathCompress = os.path.join(pathUploadCompress, 'comprimida_' + archivoNombre)
            nombreCompress = os.path.basename(pathCompress)
            nombreCompress = os.path.splitext(nombreCompress)[0] + '.jpeg'
            pathCompress = os.path.join(pathUploadCompress, nombreCompress)
            path = os.path.join(pathUpload, archivoNombre)
            if not (os.path.exists(pathUpload)):
                os.makedirs(pathUpload)
            archivo.save(path)
            comprimirImagen(path, pathCompress)
            os.remove(path)
            patente = detectar(pathCompress)
            return jsonify(
                error=patente["error"],
                errorDesc=patente["errorDesc"],
                patente=patente["patente"],
                caracteres=patente["caracteres"],
                imagen=patente["imagen"]
            )
        else:
            response = make_response("archivo no permitido", 200)
            response.mimetype = "text/plain"
            return response


def detectar(pathImg):
    global detectorNuevo
    return detectorNuevo.identificar(pathImg)


def comprimirImagen(path, pathCompress):
    imOriginal = Image.open(path)
    if imOriginal.mode in ("RGBA", "P"):
        imOriginal = imOriginal.convert("RGB")
    imOriginal.save(pathCompress, format='JPEG', quality=90)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
    import logging
    logging.basicConfig(filename='error.log', level=logging.ERROR)
