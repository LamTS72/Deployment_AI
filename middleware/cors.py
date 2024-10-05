from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app):
        app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
                allow_credentials=True,
        )