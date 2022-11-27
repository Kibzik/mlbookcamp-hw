FROM svizor42/zoomcamp-dino-dragon-lambda:v2
# Initial image contains:
#FROM public.ecr.aws/lambda/python:3.9
#COPY dino-vs-dragon-v2.tflite .

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl?raw=true

RUN pip install Pillow numpy

COPY src/lambda_function.py .

CMD ["lambda_function_hw.lambda_handler"]