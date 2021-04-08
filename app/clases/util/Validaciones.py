#!/usr/bin/python
import re


class Validaciones:

    def __init__(self):
        self.valido = False

    def validarPatente(self, patente):
        self.valido = False
        if self.__validarLngPatente(patente):
            if self.__validarFormato(patente):
                self.valido = True
        else:
            return self.valido
        return self.valido

    def __validarFormato(self, patente):
        valido = False
        if len(patente) == 6:
            regex6 = re.compile('^[A-Za-z][A-Za-z][A-Za-z][0-9][0-9][0-9]$')
            if regex6.match(patente):
                valido = True
        if len(patente) == 7:
            regex7 = re.compile('^[A-Za-z][A-Za-z][0-9][0-9][0-9][A-Za-z][A-Za-z]$')
            if regex7.match(patente):
                valido = True
        return valido

    def __validarLngPatente(self, patente):
        cTam = len(patente)
        if cTam == 6 or cTam == 7:
            validoTam = True
        else:
            validoTam = False
        return validoTam
