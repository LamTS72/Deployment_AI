import time
import sys
from pathlib import Path

from starlette.responses import Response
sys.path.append(str(Path(__file__).parent.parent))

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from utils.logger import Logger

LOGGER = Logger(__file__, log_file="http.log")

class LogMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
                start  = time.time()
                response = await call_next(request)
                process_time = time.time() - start
                # LOGGER.log.info(f"{request.client.host} - \"{request.method} {request.url.path} {request.scope["http_version"]}\" {response.status_code} {process_time}s ")
                LOGGER.log.info("{} - \"{} {} {}\" {} {}s".format(request.client.host, request.method, request.url.path, request.scope["http_version"], response.status_code, process_time))


                return response
                