# Handles file uploads in Python Tornado: http://tornadoweb.org/

import logging
import os
import uuid

import tornado.web


def uuid_naming_strategy(original_name):
    """File naming strategy that ignores original name and returns an UUID"""
    file_name, file_ext = os.path.splitext(original_name)
    return str('%s_%s.%s' % (file_name, uuid.uuid4(), file_ext))


class UploadHandler(tornado.web.RequestHandler):
    """Handle file uploads."""

    def initialize(self, upload_path, naming_strategy):
        """Initialize with given upload path and naming strategy.
        :keyword upload_path: The upload path.
        :type upload_path: str
        :keyword naming_strategy: File naming strategy.
        :type naming_strategy: (str) -> str function

        """
        self.upload_path = upload_path
        if naming_strategy is None:
            naming_strategy = uuid_naming_strategy
        self.naming_strategy = naming_strategy

    def post(self):
        fileinfo = self.request.files['filearg'][0]
        filename = self.naming_strategy(fileinfo['filename'])
        try:
            with open(os.path.join(self.upload_path, filename), 'wb') as fh:
                fh.write(fileinfo['body'])

            # or
            # file_body = self.request.files['filefieldname'][0]['body']
            # img = Image.open(StringIO.StringIO(file_body))
            # img.save("../img/", img.format)

            logging.info("%s uploaded %s, saved as %s",
                         str(self.request.remote_ip),
                         str(fileinfo['filename']),
                         filename)
            self.redirect('http://localhost:9000')
        except IOError as e:
            logging.error("Failed to write file due to IOError %s", str(e))
