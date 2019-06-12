from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect

from .models import UploadFile
from .forms import UploadFileForm, UploadFileModelForm

import cv2
import numpy as np
import logging.config
from multiprocessing.pool import ThreadPool
from .detectron import Detectron
import base64
import io
from PIL import Image
logger = logging.getLogger(__name__)


def index(request):
    uploaded_files = UploadFile.objects.all()
    context = {'uploaded_files': uploaded_files}
    return render(request, 'detectron/index.html', context)

def fs_store(file_name, file_content):
    fs = FileSystemStorage()
    filename = fs.save(file_name, file_content)
    return fs.url(filename)

def png_to_image(png):
   """Convert a png binary string to a PIL Image."""
   return Image.open(io.StringIO(png))


def image_to_png(image):
   """Convert a PIL Image to a png binary string."""
   output = io.StringIO()
   image.save(output, 'PNG')
   return output.getvalue()

def bytes_to_png(image):
   """Convert a PIL Image to a png binary string."""
   output = io.BytesIO()
   image.save(output, 'PNG')
   return output.getvalue()

def file_upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            upload_file = request.FILES.get('upload_file')
            # Need to fetch file content before FileSystemStroage.save() call
            # after save(), it become InMemoryUploadedFile
            # https://docs.djangoproject.com/en/2.2/_modules/django/core/files/uploadedfile/
            file_content = upload_file.read()  # io.BytesIO obj.
            encoded_img = cv2.imdecode(np.fromstring(file_content, np.uint8),
                                       cv2.IMREAD_UNCHANGED)
            result, marked_img = Detectron().web_run(encoded_img)

            # store uploaded file on file system at MEDIA_ROOT
            upload_file_url = fs_store(upload_file.name, upload_file)

            # form web display
            b64_src = 'data:image/png;base64,'  # png
            # b64_src = 'data:image/jpeg;charset=utf-8;base64,'  # jpg
            # original input image.
            detectron_img = b64_src + \
                base64.b64encode(file_content).decode()

            # masked image.
            d = Image.fromarray(marked_img)
            detectron_img2 = b64_src + \
                base64.b64encode(bytes_to_png(d)).decode()

            return render(request, 'detectron/file_upload.html',
                          {'form': form,
                           'upload_file_url': upload_file_url,
                           'detectron_img': detectron_img,
                           'detectron_img2': detectron_img2,})
    else:
        form = UploadFileForm()
    return render(request, 'detectron/file_upload.html', {'form': form})


def model_form_upload(request):
    if request == 'POST':
        form = UploadFileModelForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = UploadFileModelForm()
    return render(request, 'detectron/model_from_upload.html',
                  {'form': form})
