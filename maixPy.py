import sensor, image, lcd, time
import KPU as kpu

lcd.init()
lcd.rotation(2)  # Ajuste conforme necessário
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((28, 28))  # Ajuste conforme necessário
sensor.run(1)

task = kpu.load("/sd/mnist.kmodel")  # Carregue o modelo treinado

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Ajuste conforme o seu dataset

while True:
    img = sensor.snapshot()
    lcd.display(img)
    
    # Processar a imagem e realizar previsão
    img.pix_to_ai()
    fmap = kpu.forward(task, img)
    plist = fmap[:]
    pmax = max(plist)
    max_index = plist.index(pmax)
    
    # Exibir resultado
    lcd.draw_string(10, 96, "Predicted: %s" % (labels[max_index]), lcd.WHITE, lcd.BLACK)
    
kpu.deinit(task)
