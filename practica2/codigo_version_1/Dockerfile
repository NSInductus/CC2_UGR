# Elegimos la imagen base y el tag deseado.
FROM python:3.7

# Nombre y correo de la persona que mantiene el contenedor
LABEL maintainer="Angel Murcia Diaz <angelmd96@correo.ugr.es>"

# Puerto como Argumento para pasar a la hora de construir
#Por defecto sera el puerto 8080
ARG PORT=8000
EXPOSE 8000

# Establecer el directorio de trabajo
WORKDIR /

# Copiamos el archivo requirements
COPY requirements.txt /tmp/

# Instalamos lo necesario utilizando el requirements
RUN pip install --no-cache-dir -r ./tmp/requirements.txt

# Copiamos los archivos necesarios 
COPY api_v1.py /src/

# Copiamos los archivos necesarios 
COPY modeloT.pckl /src/
# Copiamos los archivos necesarios
COPY modeloH.pckl /src/

# Cambiar el directorio de trabajo
WORKDIR /src/

# Lanzar el servidor
CMD gunicorn -b 0.0.0.0:8000 api_v1:app
