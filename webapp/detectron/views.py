from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect

from .models import UploadFile
from .forms import UploadFileForm, UploadFileModelForm

import cv2


def index(request):
    uploaded_files = UploadFile.objects.all()
    # context = {'main_title': 'Detectron'}
    context = {'uploaded_files': uploaded_files}
    return render(request, 'detectron/index.html', context)


# def handle_uploaded_file(f):
#     with open('some/file/name.txt', 'wb+') as destination:
#         for chunk in f.chunks():
#             destination.write(chunk)


def file_upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            upload_file = request.FILES['upload_file']
            fs = FileSystemStorage()
            filename = fs.save(upload_file.name, upload_file)
            upload_file_url = fs.url(filename)

            processed = None

            # # case 2: opencv
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
                           'processed': processed})
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
