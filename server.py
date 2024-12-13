   from flask import Flask, render_template
   from flask_socketio import SocketIO, emit

   app = Flask(__name__)
   socketio = SocketIO(app)

   @app.route('/')
   def index():
       return render_template('webapp.html')

   @socketio.on('connect')
   def handle_connect():
       print('Client connected')

   def send_detection_data(lat, lng):
       socketio.emit('detection_update', {'lat': lat, 'lng': lng})

   if __name__ == '__main__':
       socketio.run(app, host='0.0.0.0', port=5000)