from django.db import models


def user_upload_file_path(instance, filename):
    "find the path for user upload dir."
    return 'user_{0}/{1}'.format(instance.user.id, filename)


class UploadFile(models.Model):
    desctiption = models.CharField(max_length=255, blank=True)
    upload_file = models.FileField(upload_to=user_upload_file_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
