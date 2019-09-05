from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import numpy as np
from .serializers import FileSerializer
from .recognize_faces_image import predict

import face_recognition
import argparse
import pickle
import cv2

def grab_image(stream=None):
    if stream is not None:
        data=stream.read()
    image=np.asarray(bytearray(data),dtype="uint8")
    image=cv2.imdecode(image,cv2.IMREAD_COLOR)
    return image

class FileUploadView(APIView):
    parser_class = (FileUploadParser,)
   
    def post(self, request, *args, **kwargs):
      image  = grab_image(stream=request.FILES['file'])
      post_name = predict(image)

      file_serializer = FileSerializer(data=request.data)
      if file_serializer.is_valid():
          file_serializer.save()
          return Response(post_name, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
