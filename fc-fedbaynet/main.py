from bottle import Bottle, run

from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server

from FeatureCloud.app.engine.app import app

import states
import logging

logging.getLogger("bottle").setLevel(logging.WARNING)

server = Bottle()


if __name__ == '__main__':
    app.register()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    run(server, host='localhost', port=5000, quiet=True)
