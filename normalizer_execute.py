from tools.normalize_numeric import TextNormalizerNumeric

if __name__ == '__main__':
    normalizer = TextNormalizerNumeric()
    print(normalizer.normalize("verifique si la factura F a 409516 de Hierros del Pacífico ya está apagada."))
