from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/hello')
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })


if __name__ == '__main__':
    print('starting python flask server for home price prediction...')
    app.run()