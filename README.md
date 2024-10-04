---
editor_options: 
  markdown: 
    wrap: 72
---

------------------------------------------------------------------------

# ***LABORATORIO 3 : Señales electromiográficas EMG***

------------------------------------------------------------------------

Procesamiento digital de señales

Carolina Corredor BMED B\
UMNG

------------------------------------------------------------------------

> **OBJETIVO** : Aplicar el filtrado de señales continuas para procesar
> una señal electromigráfica y detectar la fatiga muscular a través del
> análisis espectral de la misma.

------------------------------------------------------------------------

**Descripción** : Para este laboratorio se nos pide captar la señal de
la fatiga del músculo de un brazo por medio de la tecnica de EMG para
ello se hace una breve explicación de esta tecnica:

> > "El electromiograma (EMG) es una grabación de la actividad eléctrica
> > de los músculos, también llamada actividad mioeléctrica. Existen dos
> > tipos de EMG, el de superficie y el intramuscular o de aguja. Para
> > poder realizar la captura de las señales mioeléctricas se utilizan
> > dos electrodos activos y un electrodo de tierra. En el caso de los
> > electrodos de superficie, deben ser ubicados en la piel sobre el
> > músculo a estudiar, mientras que el electrodo de tierra se conecta a
> > una parte del cuerpo eléctricamente activa. La señal EMG será la
> > diferencia entre las señales medidas por los electrodos activos."

Para el desarrollo de esta guia se necesita buscar el musculo que se
piensa medir y con ello calcular la frecuencia de muestreo que es
necesaria para poder capturar la señal, realizar movimientos repetitivos
para alcanzar la fatiga del brazo y poder con estos datos realizar sus
respectivos filtros un pasa alto y un pasa bajo para eliminar las
frecuencias no deseadas .

Para el analisis de estos datos debemos dividir la señal en ventanas de
tiempo usando tecnicas como la ventana de Hamming o Hanning,
posteriormente realizar el analisis espectral de cada ventana con ayuda
de la trasnformada de Fourier (FFT) observando como cambia el espectro
de la señal conforme se acerca a la fatiga del musculo.

Por ultimo poder evaluar la dismucion de la frercuencia mediana en cada
ventana e implementar una prueba de hipotesis para verificar si el
cambio en la mediana es significativo estadisticamente .

#### ***INSTRUCCIONES***

1.Se investiga en la literatura que musculo es el mejor para poder
capturar la señal EMG para ello se diseña el circuito utilizando el
sensor AD8232,arduino,cables de electrodos ,electrodos,jumpers como se
muestra a continuacion:

![CIRCUITO](https://github.com/SeebastianOchoa/IMAGENESLAB3/blob/4efcb0c3f74646ea466c312043812b58b5261d38/CIRCUITO.jpeg)

2.Se programa el arduino para enviar los datos captados por el sensor y
asi poder ser graficados,para obtener estos datos se colocaron los
electrodos el verde y amarillo en la parte de arriba del antebrazo y el
rojo se dejo en la muñeca como tierra se le pide al usuario realizar
movimientos repetitivos hasta llegar a la fatiga del musculo mientras
estos datos son captados y guardados en un text.Como se puede observar
en la siguiente imagen.

![MONTAJE](https://github.com/SeebastianOchoa/IMAGENESLAB3/blob/4efcb0c3f74646ea466c312043812b58b5261d38/CAPTURA.jpeg)

3.Se sube este text a spyder para poder empezar con el analisis
correspondiente, lo primero que se realiza es graficar la señal y se
busca la frecuencia de muestreo adecuada utilizando el teorema de
muestreo de Nyquist donde se establece que la frecuencia de muestreo
debe ser al menos el doble de la frecuencia maxima de la señal, se
conice que las señales EMG tienen componentes de frecuencia que varian
entre 0.5Hz y 500Hz con los componentes mas importantes generalmjente
tienen un rango de 20Hz a 500 Hz, siendo asi con el teorema mencionado
anteriormeente podemos decir que la frecuencia minima de muestreo que se
manejara sera de 1000 Hz .

4.Se puede observar la grafica con los datos adquiridos para poder
analizarla de una manera mas completa, se le aplica a esta señal un
filtro pasa bajo con una frecuencia de 450 Hz eliminando los componentes
de alta frecuencia como ruido ,artefactos electronicosy un pasa alto con
una frecuencia de 20HZ eliminando los componenetes de baja frecuencia
que puede ser ruido como el movimiento d elos musculos o el ruido de
60Hz,siendo asi posible poder preservar la señal muscular limpia. Para
una mejor comprension se le aplico el ventanamiento Hanning esta es una
tecnica que se utiliza para suavizar los efectos indeseados en el
analisis espectral ademas es imporatbnte resaltar que nos permirte
trabajar en el dominio de la frecuencia para poder aplicarle la
transformada rapida de (FFT) pero antes de realizar este paso ,los picos
que se observan de frecuencia resultantes son más representativos de las
verdaderas componentes de la señal. El ventanado minimiza el ruido que
resulta de las discontinuidades. También mejora la resolución de
frecuencia de los picos, haciéndolos más nítidos y permitiendo una mejor
identificación de las frecuencias dominantes en la señal Obteniendo asi
grandes ventajas como lo son la reduccion del ruido espectral,minimiza
la fuga espectral mejorando la separacion de los componentes de la
frecuencia cercana ya que atenua los extremos de cada segmento

Se visualiza en la siguiente imagen las graficas la señal original y el
ventamiento Hanning.

```         
# Importamos las librerías necesarias
import numpy as np  # Para operaciones numéricas y manejo de arrays
import matplotlib.pyplot as plt  # Para graficar
from scipy.signal import butter, filtfilt, windows  # Para filtros y ventanas
from scipy.fft import fft, fftfreq  # Para transformadas de Fourier

# Función para diseñar un filtro Butterworth
def butter_filter(data, lowcut, highcut, fs, btype='low', order=4):
    nyq = 0.5 * fs  # Frecuencia de Nyquist

    # Determinamos los coeficientes del filtro dependiendo del tipo
    if btype == 'low':
        normal_cut = highcut / nyq  # Normalizamos la frecuencia de corte
        b, a = butter(order, normal_cut, btype=btype, analog=False)  # Diseñamos el filtro
    elif btype == 'high':
        normal_cut = lowcut / nyq  # Normalizamos la frecuencia de corte
        b, a = butter(order, normal_cut, btype=btype, analog=False)  # Diseñamos el filtro
    elif btype == 'band':
        normal_lowcut = lowcut / nyq  # Normalizamos la frecuencia de corte inferior
        normal_highcut = highcut / nyq  # Normalizamos la frecuencia de corte superior
        b, a = butter(order, [normal_lowcut, normal_highcut], btype='band')  # Diseñamos el filtro
    else:
        raise ValueError("Tipo de filtro no soportado")  # Manejo de errores para tipos de filtro no válidos
        
    y = filtfilt(b, a, data)  # Filtramos la señal sin fase
    return y  # Retornamos la señal filtrada

# Función para dividir la señal en segmentos
def segmentar_senal(data, segmento_len):
    num_segmentos = len(data) // segmento_len  # Calculamos el número de segmentos
    return np.array_split(data, num_segmentos)  # Dividimos la señal en segmentos

# Función para calcular la FFT de un segmento y la frecuencia mediana
def calcular_fft(segmento, fs):
    ventana_hanning = windows.hann(len(segmento))  # Creamos una ventana de Hanning
    segmento_aventanado = segmento * ventana_hanning  # Aplicamos la ventana al segmento
    fft_segmento = fft(segmento_aventanado)  # Realizamos la FFT del segmento
    freqs = fftfreq(len(segmento_aventanado), 1 / fs)  # Calculamos las frecuencias correspondientes
    
    # Retornar las frecuencias y la magnitud de la FFT
    magnitud = np.abs(fft_segmento[:len(fft_segmento)//2])  # Magnitud de la FFT (solo parte positiva)
    freqs = freqs[:len(freqs)//2]  # Frecuencias correspondientes (solo parte positiva)
    
    # Calcular la frecuencia mediana
    frecuencia_mediana = freqs[np.argmax(magnitud)]  # Frecuencia donde la magnitud es máxima
    
    return freqs, magnitud, frecuencia_mediana  # Retornamos frecuencias, magnitud y frecuencia mediana

# Leer los datos del archivo
with open('datos.txt', 'r', encoding='latin1') as file:
    lineas = file.readlines()  # Leemos todas las líneas del archivo

# Convertir las líneas a un array de números
datos = np.array([float(line.strip()) for line in lineas if line.strip().replace('.', '', 1).isdigit()])  # Convertimos las líneas a float


# Parámetros
fs = 1000  # Frecuencia de muestreo en Hz
lowcut = 20.0  # Frecuencia de corte para el filtro pasa altas
highcut = 450.0  # Frecuencia de corte para el filtro pasa bajas
segmento_len = int(1 * fs)  # Longitud de cada segmento en tiempo (1 segundo)
order = 6  # Orden del filtro para hacerlo más limpio

# Filtrado pasa altas
datos_filtrados_altas = butter_filter(datos, lowcut, highcut, fs, btype='high', order=order)  # Aplicamos filtro pasa altas

# Filtrado pasa bajas
datos_filtrados_bajas = butter_filter(datos_filtrados_altas, lowcut, highcut, fs, btype='low', order=order)  # Aplicamos filtro pasa bajas

# Filtrado pasa banda (con mayor filtrado para dejar la señal más limpia)
datos_filtrados_band = butter_filter(datos, lowcut, highcut, fs, btype='band', order=order)  # Aplicamos filtro pasa banda

# Graficar la señal original y las señales filtradas
plt.figure(figsize=(12, 12))  # Establecemos el tamaño de la figura

# Señal original
plt.subplot(5, 1, 1)  # Creamos el primer subplot
plt.title('Señal EMG Original')  # Título del subplot
plt.plot(datos)  # Graficamos la señal original
plt.grid(True)  # Activamos la cuadrícula

# Señal filtrada (pasa altas)
plt.subplot(5, 1, 2)  # Creamos el segundo subplot
plt.title('Señal Filtrada (Pasa Altas)')  # Título del subplot
plt.plot(datos_filtrados_altas, color='orange')  # Graficamos la señal filtrada con color naranja
plt.grid(True)  # Activamos la cuadrícula

# Señal filtrada (pasa bajas)
plt.subplot(5, 1, 3)  # Creamos el tercer subplot
plt.title('Señal Filtrada (Pasa Bajas)')  # Título del subplot
plt.plot(datos_filtrados_bajas, color='green')  # Graficamos la señal filtrada con color verde
plt.grid(True)  # Activamos la cuadrícula

# Señal filtrada (pasa banda)
plt.subplot(5, 1, 4)  # Creamos el cuarto subplot
plt.title('Señal Filtrada (Pasa Banda) - Más limpia')  # Título del subplot
plt.plot(datos_filtrados_band, color='purple')  # Graficamos la señal filtrada con color púrpura
plt.grid(True)  # Activamos la cuadrícula
```

![GRAFICA DE FILTROS](https://github.com/SeebastianOchoa/IMAGENESLAB3/blob/4efcb0c3f74646ea466c312043812b58b5261d38/FILTROS2.jpeg)

Podemos analizar lo siguiente de las graficas la señal EMG original
muestra la actividad eléctrica del músculo durante el movimiento. Se
pueden observar patrones de activación muscular, con picos que
corresponden a contracciones y relajaciones del músculo.se observa que
puede haber ruido y artefactos, lo cual es normal en las señales EMG. Es
importante notar si la señal original tiene un fondo de ruido
significativo que podría afectar el análisis posterior.

para la **señal filtrada (Pasa Altas)**.Este gráfico muestra la señal
después de aplicar un filtro pasa-altas, que elimina las frecuencias
bajas y permite observar mejor los cambios rápidos en la actividad
muscular. La señal se vuelve más "aguda", con picos más definidos. Esto
puede ayudar a destacar la actividad muscular rápida, pero puede no
capturar completamente las contracciones sostenidas de baja frecuencia.
La **señal filtrada (Pasa Bajas)**: Este gráfico presenta la señal tras
aplicar un filtro pasa-bajas, que elimina las altas frecuencias y
suaviza la señal.La señal ahora es más suave y más fácil de interpretar.
Esto puede ser útil para observar tendencias generales en la activación
muscular, pero puede perder información importante sobre contracciones
rápidas.

La **señal filtrada (Pasa Banda)**se observa el resultado de un filtrado
pasa-banda, que retiene las frecuencias de interés para la actividad
muscular (20-450 Hz en nuestro caso ).Esta es probablemente la señal más
informativa, ya que elimina tanto el ruido de baja frecuencia como las
frecuencias muy altas. Se puede observar cómo la actividad muscular
varía a lo largo del tiempo.

```         
# Aplicar ventana de Hanning a la señal completa para mostrar
aventanamiento ventana_hanning_completa =
windows.hann(len(datos_filtrados_band)) \# Creamos una ventana de
Hanning para la señal completa datos_aventanados = datos_filtrados_band
\* ventana_hanning_completa \# Aplicamos la ventana a la señal filtrada
plt.subplot(5, 1, 5) \# Creamos el quinto subplot plt.title('Señal
Aventanada (Ventana de Hanning)') \# Título del subplot
plt.plot(datos_aventanados, color='red') \# Graficamos la señal
aventanada con color rojo plt.grid(True) \# Activamos la cuadrícula

plt.tight_layout() \# Ajustamos el layout para que no se superpongan
plt.show() \# Mostramos las gráficas

# Segmentar la señal filtrada en intervalos de tiempo

segmentos = segmentar_senal(datos_filtrados_band, segmento_len) \#
Segmentamos la señal filtrada en intervalos de tiempo

# Graficar la FFT para cada segmento y calcular la frecuencia mediana

plt.figure(figsize=(12, 12)) \# Establecemos el tamaño de la figura
frecuencias_medianas = [] \# Inicializamos la lista para almacenar
frecuencias medianas

# Iteramos sobre cada segmento

for i, segmento in enumerate(segmentos): freqs, fft_magnitud,
frecuencia_mediana = calcular_fft(segmento, fs) \# Calculamos la FFT y
la frecuencia mediana frecuencias_medianas.append(frecuencia_mediana) \#
Añadimos la frecuencia mediana a la lista

        
# Graficar la FFT de cada segmento
plt.subplot(len(segmentos), 1, i + 1)  # Creamos un subplot para cada segmento
plt.plot(freqs, fft_magnitud)  # Graficamos la magnitud de la FFT
plt.title(f'Transformada de Fourier - Segmento {i + 1}')  # Título del subplot
plt.xlabel('Frecuencia [Hz]')  # Etiqueta del eje x
plt.ylabel('Magnitud')  # Etiqueta del eje y
plt.grid(True)  # Activamos la cuadrícula


plt.tight_layout() \# Ajustamos el layout para que no se superpongan
plt.show() \# Mostramos las gráficas
```

**Señal Aventanada** tomamos la ventana de Hanning donde se muestra la
señal filtrada después de aplicar una ventana de Hanning, que reduce los
efectos de fuga en la FFT.La aplicación de la ventana hace que la señal
sea más adecuada para el análisis en frecuencia. El uso de una ventana
puede ayudar a mejorar la claridad en el espectro de frecuencia
posterior.

Al observar las gráficas, es importante analizar cómo cambia la amplitud
de la señal a medida que el músculo se fatiga. En general, con la
fatiga, es probable que la señal EMG muestre una disminución en la
amplitud y un aumento en la variabilidad.

![GRAFICAS CON AVENTANIAMIENTO HANNING](https://github.com/SeebastianOchoa/IMAGENESLAB3/blob/4efcb0c3f74646ea466c312043812b58b5261d38/FILTROS1.jpeg)

**Análisis de la Transformada de Fourier**

El eje X representa la frecuencia (en Hz), mientras que el eje Y muestra
la magnitud de cada componente de frecuencia en la señal.Se observa que
las frecuencias de interés para la actividad muscular normalmente se
encuentran en un rango de 20 a 450 Hz.

**Segmento 1**: Presenta un pico bastante pronunciado en bajas
frecuencias, lo que indica una fuerte presencia de componentes en esa
área.Si el pico está en un rango esperado, esto puede reflejar una
activación muscular clara. Sin embargo, si la amplitud es excesivamente
alta, podría sugerir ruido o artefactos en la señal.

**Segmento 2**:Muestra un pico menos pronunciado que el segmento 1. Esto
podría indicar una menor actividad muscular, posiblemente porque el
músculo está comenzando a fatigarse o porque se ha reducido la
intensidad de la contracción.

**Segmento 3**:Vuelve a mostrar un pico significativo, similar al
segmento 1. La presencia de un pico en esta parte de la señal puede
indicar un aumento momentáneo en la actividad muscular, quizás debido a
un esfuerzo adicional. Si hay mucha variabilidad en la forma del pico,
puede ser un indicativo de que se están presentando contracciones no
uniformes.

**Segmento 4**:Presenta un pico similar a los anteriores, pero con menor
magnitud.Esto podría ser un signo claro de fatiga muscular. Una
disminución en la magnitud sugiere que el músculo ya no está produciendo
la misma cantidad de energía o actividad eléctrica.

La variabilidad en la magnitud de los picos entre los segmentos puede
ofrecer información sobre cómo la fatiga muscular afecta la actividad
eléctrica. La señal debería ser interpretada junto con otros
indicadores, como el tiempo que se ha estado realizando la tarea y las
repeticiones.

Si hay un pico muy alto que no corresponde a la actividad muscular,
podría indicar la presencia de ruido en la señal. Esto es crítico, ya
que puede afectar la interpretación de los resultados. Un ruido elevado
en la frecuencia puede hacer que los resultados sean menos confiables.

![GRAAFICAS ANALISIS ESPECTRAL](https://github.com/SeebastianOchoa/IMAGENESLAB3/blob/4efcb0c3f74646ea466c312043812b58b5261d38/ANALISIS%20ESPECTRAL.jpeg)

```         

# Mostrar frecuencias medianas

print("Frecuencias medianas de cada segmento:") \# Mensaje de inicio for
i, f_median in enumerate(frecuencias_medianas): print(f"Segmento {i +
1}: {f_median:.2f} Hz") \# Imprimimos la frecuencia mediana de cada
segmento
```
![FRECUENCIAS MEDIANAS](https://github.com/SeebastianOchoa/IMAGENESLAB3/blob/4efcb0c3f74646ea466c312043812b58b5261d38/FRECUENCIA%20MEDIA.jpeg)


las **frecuencias medianas** calculadas son coherentes con el rango
esperado para la actividad muscular, esto indica que los filtros están
funcionando correctamente. Durante movimientos repetitivos, el ruido
puede aumentar, lo que puede hacer que las frecuencias mediadas cambien.
Observa si el filtrado ha reducido este ruido y si los picos permanecen
en el rango normal de actividad muscular.

Las gráficas reflejan adecuadamente los cambios en la actividad muscular
y el efecto de los filtros aplicados.

Las frecuencias entre 50 y 60 Hz son típicas para la actividad muscular.
Esto puede incluir la activación de fibras musculares motoras,
especialmente durante contracciones isotónicas o isométricas. Estos
valores son consistentes y no muestran grandes desviaciones, lo cual es
bueno.

La similitud en las frecuencias entre los segmentos sugiere que, aunque
puede haber variabilidad en la amplitud de la señal, la frecuencia de
activación muscular se mantiene constante. Esto puede indicar que el
patrón de activación del músculo se ha mantenido a lo largo de la
actividad, lo que es un signo positivo en cuanto a la eficiencia de la
función muscular.

La fatiga muscular se espera observar disminución en la amplitud de los
picos puede disminuir debido a la fatiga, lo que se observa en tus
gráficos. En algunos estudios, se ha reportado que durante la fatiga,
puede haber un desplazamiento hacia frecuencias más altas o bajas
dependiendo de la dinámica de la contracción y el tipo de músculo. Sin
embargo, en tu caso, la frecuencia se ha mantenido bastante constante,
lo cual puede ser un indicativo de que la fatiga no ha alterado
drásticamente la activación neuromuscular, aunque hay que considerar
otros factores.

\`
