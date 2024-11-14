# AREP - Taller LLMChatGPTliveParte2
## Autor: Cristian David Naranjo Orjuela

El propósito de este taller es desarrollar una aplicación web básica para la traducción de texto utilizando LangChain y un modelo de lenguaje LLM.

## prerrequisitos
* Git - Control de versiones.
* python - Lenguaje de programación.
* OPENAI_API_KEY - Debido a temas de seguridad no es posible alojar la llave en el repositorio, por eso debe contar con una para reemplazarla en el codigo para que funcione.
* Jupyter - Entorno interactivo que permite la creación de documentos que contienen código y texto explicativo.


## Versiones 
Python 3.12.5

Visual Studio Code: 1.95.1


### Instalación y Ejecución
Para ejecutar la aplicación es necesario instalar Python 3.12.7 o versiones similares y git. El primer paso es clonar el repositorio e ingresar a la carpeta resultante

```
https://github.com/cristiandavid0124/RagChatGpt2parteArem.git
````

en el archivo del server deve poner la clave que asigno el profe en este caso quitele la frase " holaaaaa "al inicio de la llave debido  que github no deja colocarla



```
os.environ["OPENAI_API_KEY"] = "holaaaaassk-proj-AwftCWlW0Ksr6jMYxnN7SSLrJMrB6_jV5jqt_ue0vgyxE34dHzaTVzYlG9B5BHi25jXBDs4swQT3BlbkFJOA1Ej8m-_XWXwBFLSYVvTpmWNQGCZDIfhw7gVG-JDpQ4HStL19RaGCceuHNsp9kkRWdQ3-iHsA"
```

Luego ingrese los comandos :

```
python -m venv .venv
```

```
.venv\Scripts\activate
```

```
pip install -r requirements.txt
```

luego de la instalacion puede ejecutar el servidor con:

```
python RAGServer.py
```

Finalmente pruebe el servidor ingresando a http://localhost:8000/LangChain?pregunta=digame comunidades de videojuego en Cuba

la ia fue entrenada con articulos de videojuegos

![image](https://github.com/user-attachments/assets/4f08affb-8f07-4773-9b37-2a678cc6bb1f)





### Arquitectura

La aplicación se basa en una arquitectura que emplea FastAPI y LangChain para implementar un modelo de Generación Aumentada por Recuperación (RAG). FastAPI funciona como el servidor web que expone un único endpoint, /rag, a través del cual los usuarios pueden enviar preguntas y recibir respuestas generadas por el sistema. Esta API permite a los usuarios realizar consultas sobre temas específicos, obteniendo respuestas relevantes y contextualizadas.

El núcleo de la aplicación es LangChain, una biblioteca que facilita la integración de modelos de lenguaje en flujos complejos de procesamiento de texto. La aplicación usa Chroma, una base de datos vectorial, para almacenar y recuperar documentos relacionados con la consulta del usuario. Estos documentos se obtienen de URLs específicas mediante WebBaseLoader y se dividen en partes más pequeñas con RecursiveCharacterTextSplitter para indexarlos. Los fragmentos de texto se convierten en vectores mediante OpenAIEmbeddings, lo que permite una recuperación eficiente de la información.

Cuando un usuario realiza una consulta, la pregunta se analiza junto con los documentos recuperados, utilizando GPT-4 (mediante ChatOpenAI) para elaborar una respuesta. LangChain conecta todos estos componentes en un flujo que abarca la recuperación de documentos, generación de texto y posprocesamiento de la respuesta con un StrOutputParser. Toda esta estructura es gestionada por una cadena RAG, permitiendo que el modelo aproveche la información recuperada para dar respuestas más precisas y contextualizadas


![diagrama de arquitectura](https://github.com/user-attachments/assets/3f60d22b-c64a-45cd-acaf-b8adcd6532f0)


### Funcionamiento

pregunta  = digame comunidades de videojuego en Cuba

![image](https://github.com/user-attachments/assets/32727563-e02f-42ab-b006-55bfe74ed671)

Pregunta = Los videojuegos por que  son importantes 

![image](https://github.com/user-attachments/assets/aa9b7006-e50e-4714-8b6e-3233b71e174e)

Ahora pregunta no relacionada con lo aprendido de la ia

Pregunta = Como va Colombia en Las eliminatorias

![image](https://github.com/user-attachments/assets/181f867b-1f76-4dca-80f7-2c8442adda94)



## Agradecimientos
* Al profesor Daniel Benavides por impartir esta clase y tener la pasión de enseñar.


