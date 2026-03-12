# DETECCIÓN DE INCENDIOS EN INTERIORES EN TIEMPO REAL MEDIANTE VISIÓN POR COMPUTADORA UTILIZANDO REDES NEURONALES CONVOLUCIONALES

Este repositorio contiene el código y los recursos para el trabajo final de grado.

La investigación fue desarrollada entre **2025 y 2026**.

![repoGif2](https://github.com/user-attachments/assets/d0110a75-1e2d-4516-9b0d-1046d9b37128)

---

## 📌 Descripción del Proyecto

Esta investigación propone un enfoque basado en **visión por computadora**, utilizando cámaras y algoritmos de procesamiento de imágenes para detectar patrones visuales asociados al fuego o humo, permitiendo así una **detección más temprana**.

---

## 🎯 Objetivos

### Objetivo General

Detectar incendios en interiores en tiempo real mediante visión por computadora utilizando redes neuronales convolucionales.

### Objetivos Específicos

-	Construir un dataset compuesto por imágenes y videos de incendios en interiores.
-	Entrenar el modelo para la detección de incendios.
-	Validar el modelo mediante métricas que evalúen su desempeño.
-	Integrar el modelo entrenado a un sistema de monitoreo en tiempo real utilizando una cámara de circuito cerrado.

---

## ⚙️ Tecnologías Utilizadas

- Python (OpenCv, PyTorch, Flask, Ultralytics)
- YOLO
- Label Studio
- MediaMTX
- FFmpeg
---

## 📊 Resultados


Los resultados obtenidos durante la evaluación del sistema desarrollado demostraron un desempeño satisfactorio en la detección temprana de incendios. 
El modelo es capaz de identificar el 100 % de los eventos de incendio evaluados, ya sea a través de la detección de humo o mediante la identificación directa de fuego. 
En particular, la detección de fuego presentó el mejor rendimiento, logrando identificar llamas de pequeño tamaño de manera precisa y consistente, incluso en escenarios con bajo contraste o iluminación variable.

<img width="836" height="484" alt="image" src="https://github.com/user-attachments/assets/aa70af7d-b33b-479f-95fb-c6be55deb14b" />

En cuanto a la detección efectiva del evento, el tiempo requerido para que el sistema identifique un incendio desde su inicio oscila entre 1 segundo y 30 segundos como máximo, dependiendo principalmente de las condiciones visuales del humo o del fuego presentes en la escena. 
Este rango de tiempo resulta adecuado para escenarios de vigilancia y detección temprana, donde la rapidez de respuesta es un factor crítico.

<img width="831" height="308" alt="image" src="https://github.com/user-attachments/assets/85421d0f-4113-4e09-b0ca-02b4cbe65d22" />

El sistema es capaz de realizar alertas sonoras y tambien mediante servicios de mensajería

<img width="547" height="119" alt="image" src="https://github.com/user-attachments/assets/723fb5d6-688d-4b77-97e2-d8f3b13274dc" />
<img width="598" height="411" alt="image" src="https://github.com/user-attachments/assets/ddccaafa-9164-4f05-b22d-0b7a35cf0ba8" />

## 📄 Libro de la tesis

El libro completo de la investigación puede consultarse en el siguiente enlace:

🔗 **[Ver tesis completa](https://drive.google.com/file/d/1LSVUsA1KwM0FYvQrtBRsrHP0yliC5J_p/view?usp=sharing)**

## 👨‍💻 Autor

**Carlos Lugo**

Trabajo desarrollado como parte de una tesis de grado realizada entre **2025 y 2026** para la obtención del título de Ingeniero Informático.




