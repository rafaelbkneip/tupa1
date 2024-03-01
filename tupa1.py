import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Definir variáveis
hmin, smin, vmin = 0, 0, 228
hmax, smax, vmax = 179, 98, 255
area = 0
cont_frame = 0

# Inicializar PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)  # Permitir que a câmera inicialize

# Loop para processar cada frame da PiCamera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Obter o array de imagem do frame
    imagem = frame.array

    # Transformar a imagem em escala HSV
    imgHSV = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Parâmetros para a máscara
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])
    
    # Aplicar máscara
    máscara = cv2.inRange(imgHSV, lower, upper)
    
    # Localizar contornos na imagem
    contornos, hierarchy = cv2.findContours(máscara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Inicializar variável para desenho na imagem
    desenho = np.zeros_like(imagem)

    # Verificar se existem contornos na imagem
    if len(contornos) == 0:
        print("No frame", cont_frame, "nenhum raio foi detectado.")
        
    # Caso existam contornos, processá-los
    else:
        # Definir o maior contorno
        aux_area = 0
        for i, contorno in enumerate(contornos):
            area = cv2.contourArea(contorno)
            if area >= aux_area:
                aux_area = area
                maior_contorno = i

        # Calcular o centróide da maior área
        M = cv2.moments(contornos[maior_contorno])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Desenhar o contorno e centróide na imagem
        color = (0, 0, 255)
        cv2.drawContours(desenho, contornos, maior_contorno, color, 1, 8, hierarchy, 0)
        cv2.circle(desenho, (cx, cy), 4, color, -1, 8, 0)
        
        # Exibir informações sobre o maior contorno
        if aux_area > 0:
            print("No frame", cont_frame, "o maior raio possui uma área igual a", aux_area, ", com centróide localizado nas coordenadas", (cx, cy), ".")
        else:
            print("No frame", cont_frame, "nenhum raio foi detectado.")
    

    # Limpar o buffer de captura para o próximo frame
    rawCapture.truncate(0)
    
    # Avançar para o próximo frame
    cont_frame += 1

    # Condição para interromper o loop (por exemplo, após um certo número de frames)
    if cont_frame >= 100:
        break

# Finalizar a PiCamera
camera.close()
cv2.destroyAllWindows()