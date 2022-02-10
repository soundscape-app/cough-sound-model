from waitress import serve
from predict import app

serve(app, host='0.0.0.0', port=5000)