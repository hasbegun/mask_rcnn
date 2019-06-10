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

logger = logging.getLogger(__name__)
# logger = logging.config.dictConfig()


def index(request):
    uploaded_files = UploadFile.objects.all()
    # context = {'main_title': 'Detectron'}
    context = {'uploaded_files': uploaded_files}
    return render(request, 'detectron/index.html', context)


# def handle_uploaded_file(f):
#     with open('some/file/name.txt', 'wb+') as destination:
#         for chunk in f.chunks():
#             destination.write(chunk)


def fs_store(file_name, file_content):
    fs = FileSystemStorage()
    filename = fs.save(file_name, file_content)
    return fs.url(filename)


def file_upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            upload_file = request.FILES.get('upload_file')
            # Need to fetch file content before FileSystemStroage.save() call
            # after save(), it become InMemoryUploadedFile
            # https://docs.djangoproject.com/en/2.2/_modules/django/core/files/uploadedfile/
            file_content = upload_file.read()  # io.BytesIO obj.
            # fs = FileSystemStorage()
            # filename = fs.save(upload_file.name, upload_file)
            # upload_file_url = fs.url(filename)

            decoded_img = cv2.imdecode(np.fromstring(file_content, np.uint8),
                                       cv2.IMREAD_UNCHANGED)
            # result = Detectron().web_run(decoded_img)

            # store uploaded file on file system at MEDIA_ROOT
            upload_file_url = fs_store(upload_file.name, upload_file)

            # fs = FileSystemStorage()
            # filename = fs.save(upload_file.name, upload_file)
            # upload_file_url = fs.url(filename)

            # form web display
            b64_src = 'data:image/png;base64,'  # jpeg
            detectron_img = b64_src + \
                base64.b64encode(decoded_img).decode()

            import pdb
            pdb.set_trace()

            # # case 2: opencv
            # fileinfo = self.request.files['filename'][0]
            # filename = self.naming_strategy(fileinfo['filename'])

            # decoded_img = cv2.imdecode(numpy.fromstring(fileinfo['body'], numpy.uint8),
            #                 cv2.IMREAD_UNCHANGED)
            # # detect_info = MaskRCNNWraper().web_run(decoded_img)

            # # show original image
            # b64_src = 'data:image/png;base64,'  #jpeg
            # img_src = b64_src + base64.b64encode(fileinfo['body']).decode()
            # self.render("display_img.html", filename=img_src)

            # Thread(target=self.__save_uploaded_file, args=(filename, decoded_img)).start()
            # self.redirect('http://localhost:9000')

            return render(request, 'detectron/file_upload.html',
                          {'form': form,
                           'upload_file_url': upload_file_url,
                           'detectron_img': detectron_img})
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
