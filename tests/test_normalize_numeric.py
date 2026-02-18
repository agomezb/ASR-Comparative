#!/usr/bin/env python3
import unittest
import sys
import os

# Agregamos el directorio 'tools' al path para poder importar el módulo
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from normalize_numeric import TextNormalizerNumeric

class TestTextNormalizerNumeric(unittest.TestCase):
    def setUp(self):
        self.normalizer = TextNormalizerNumeric()

    def test_basic_numbers(self):
        """Prueba la conversión básica de palabras a números (cinco -> 5)"""
        self.assertEqual(self.normalizer.normalize("cinco"), "5")
        self.assertEqual(self.normalizer.normalize("diez"), "10")
        self.assertEqual(self.normalizer.normalize("cincuenta"), "50")

    def test_technical_replacements(self):
        """Prueba reemplazos técnicos como i7, a4, fa"""
        self.assertEqual(self.normalizer.normalize("i 7"), "i7")
        self.assertEqual(self.normalizer.normalize("a 4"), "a4")
        self.assertEqual(self.normalizer.normalize("f a"), "fa")
        self.assertEqual(self.normalizer.normalize("1tb"), "1 terabyte")

    def test_oov_replacements(self):
        """Prueba correcciones de palabras fuera de vocabulario (OOV)"""
        self.assertEqual(self.normalizer.normalize("compu facil"), "compufacil")
        self.assertEqual(self.normalizer.normalize("tecno sis"), "tecnosys")

    def test_numeric_sequences_unification(self):
        """Prueba la unificación de secuencias numéricas (80 20 25 -> 802025)"""
        self.assertEqual(self.normalizer.normalize("80 20 25"), "802025")
        self.assertEqual(self.normalizer.normalize("80-20-25"), "802025")
        self.assertEqual(self.normalizer.normalize("1 2 3"), "123")

    def test_codes_separation(self):
        """Prueba la separación de letras y números en códigos complejos"""
        # fa4095 -> fa 4095
        self.assertEqual(self.normalizer.normalize("fa4095"), "fa 4095")
        
        # Caso combinado: "f a 4095" -> "fa 4095"
        self.assertEqual(self.normalizer.normalize("f a 4095"), "fa 4095")

    def test_punctuation_removal(self):
        """Prueba la eliminación de puntuación"""
        self.assertEqual(self.normalizer.normalize("hola_mundo"), "hola mundo")
        self.assertEqual(self.normalizer.normalize("auto-matico"), "auto matico") 

if __name__ == '__main__':
    unittest.main()
