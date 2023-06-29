# import requirements needed
from flask import Flask, render_template, redirect, url_for, session, flash, request
from utils import get_base_url
from flask import request as flask_request
import requests
import base64
import os
import shutil
from werkzeug.utils import secure_filename
# import bbox_visualizer as bbv
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# from roboflow import Roboflow

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
#clears uploads folder on flask app run
for filename in os.listdir(UPLOAD_FOLDER):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))



# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')


# adds upload folder to base app directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(64)

# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('index1.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route(f'{base_url}', methods=['POST'])
# def upload_file():
#   print("uploading?")
#   print("reached to upload section")
#   if request.method == 'POST':
#     print("reached to post method")
#     if 'file' not in request.files:
#       print(request.files)
#       print("reached to file found")
#       flash('no file part')
#       return redirect(request.url)

#     file = request.files['file']

#     if file.filename == '':
#       print("reached to file not present")
#       flash("no selected file")
#       return redirect(request.url)

#     if file and allowed_file(file.filename):
#       print("reached to file present")
#       filename = secure_filename(file.filename)
#       file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#       filename = 'uploads/'+filename
#       print('filename: ', filename)
#       result = query(filename)
#       print('result: ', result)
#       return render_template('results.html')

@app.route(f'{base_url}', methods=['POST'])
def upload_file():
  
  try:
    # rec_type = "new val"

    print("uploading?")
    # session["data"] = "genre_guess_label"
    # rec_type = "genre_guess_label"
    print("reached to upload section")
    if flask_request.method == 'POST':
      print("reached to post method")
      if 'file' not in flask_request.files:
        print("reached to file found")
        flash('no file part')
        return redirect(flask_request.url)

      file = flask_request.files['file']

      if file.filename == '':
        print("reached to file not present")
        flash("no selected file")
        return redirect(flask_request.url)

      if file and allowed_file(file.filename):
        print("reached to file present")
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        with open(f'uploads/{filename}', 'rb') as image_file:
          image_base64 = image_file.read()
          image = base64.b64encode(image_base64).decode('utf-8')

          url = "https://detect.roboflow.com/outsident/1"
          params = {"api_key": "V0EgyW40xJS7sGZUm9SH"}
          headers = {"Content-Type": "application/x-www-form-urlencoded"}

          response = requests.post(url,
                                   params=params,
                                   data=image,
                                   headers=headers)

          if response.status_code == 200:

            result = response.json()
            print(result)
            print('filename', filename)
            print(response)

            
            print('before bbox')
            bbox_and_save(filename, result, 'static/assets/img/test/result.png')
            print('after bbox')
            # print(result['predictions'][0]['class'])
            # i = 0
            # if len(result['predictions'])==0:
            #   pred_class = "Nothing was found, Percent confidence indeterminate"
            # else:
            #   pred_class = ""
            #   for i in range(len(result['predictions'])):
            #     pred_class = pred_class + str(result['predictions'][i]['class']) + " found with " + str(round(float(result['predictions'][i]['confidence'])*100,2)) + " percent confidence!\n"
            # print(pred_class)
            pred_class=[]
            rtn=""
            rtnWp = []
            for i in range(len(result['predictions'])):
              pred_class.append(str(result['predictions'][i]['class']))
              
              # + " found with " + str(round(float(result['predictions'][i]['confidence'])*100,2)) + "% confidence!"
            if len(pred_class) == 0:
              rtn = "No Threat Detected :)"
            else:
              rtn = "Threat Detected!"
              for i in range(len(pred_class)):
                temp = pred_class[i].capitalize() +': ' + str(pred_class.count(pred_class[i]))
                if temp not in rtnWp:
                  rtnWp.append(temp)
    
            print(rtnWp)
              
            # print("ran")
            
            return redirect(url_for('results', wList=rtnWp, rtn=rtn))
          else:
            # session["data"] = "genre_guess_label"
            print("reached to api error")
            print("API error occurred:", response.text)
            flash(
              'API error has occurred, wait a few seconds and try again...')
            return redirect(url_for('home'))
  except:
    return redirect(url_for('home'))



@app.route(f'{base_url}/results')
def results():
  dng = request.args.get('rtn') 
  wList = request.args.getlist('wList')
  return render_template('results.html', dng=dng, wList=wList)

# # define additional routes here
# # for example:
# # @app.route(f'{base_url}/team_members')
# # def team_members():
# #     return render_template('team_members.html') # would need to actually make this page

def query(filename):
  with open(filename, "rb") as image_file:
    
    image_data = image_file.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

  url = "https://detect.roboflow.com/outsident/1"
  params = {
    "api_key": "V0EgyW40xJS7sGZUm9SH"
            }
  headers = {
    "Content-Type": "application/x-www-form-urlencoded"
            }
  data = {
    "data": base64_image
   }

  response = requests.post(url, params=params, data=data, headers=headers)
  # return response.json()  
  if response.status_code == 200: 
    return response.json()
  else:
    return response.text

def bbox_and_save(image_path, response_data, output_path=None):
  
    # Load the image
  print('before open')
  image_path='uploads/'+image_path
  image_data = Image.open(image_path).convert('L')
  print('image path: ',image_path)
  
  # image = Image.open(image_path)
  # image_data = plt.imread(image_path)
  # with open(image_path, "rb") as image_file:
  #   image_data = image_file.read()
  print('after open')

    # Load JSON data
    # with open(json_path) as json_file:
    #     data = json.load(json_file)
  data = response_data

    # Get the bounding box information
  predictions = data['predictions']
  print('before draw')

  fig, ax = plt.subplots()

    # Display the image
  ax.imshow(image_data, cmap='gray', vmin=0, vmax=255)

    # Draw bounding boxes on the image
  # data['image']['width']
  # data['image']['height']
  for prediction in predictions:
    width = int(prediction['width'])
    height = int(prediction['height'])
    x = int(prediction['x']) - width/2
    y = int(prediction['y']) - height/2

    # curr_class = prediction['class']
    

        # Create a rectangle patc
    rect = plt.Rectangle((x, y), width, height, edgecolor= "#d9bf82", facecolor='none', linewidth=2)
    print('after rect')

        # Add the rectangle patch to the axis
    ax.add_patch(rect)
    print('after add')
    label_x = x  # Adjust the x-coordinate for the label position
    label_y = y - 15  # Adjust the y-coordinate for the label position
    print(prediction['class'])
    ax.text(label_x, label_y, prediction['class'] + ' ' + str(prediction['confidence'])[:4], ha='left', va='top', fontsize=8,
        bbox=dict(facecolor="#d9bf82", edgecolor="#d9bf82", boxstyle='square,pad=0.1'))  # Set ha='left' and va='top' for top left position. 
    print('after text')

    # Save or show the image with bounding boxes
  if output_path:
    plt.savefig(output_path)




    # Create a drawing object
  # draw = ImageDraw.Draw(image_data)
  # print('after draw')
    # Draw bounding boxes on the image
  # for prediction in predictions:
  #   x = int(prediction['x'])
  #   y = int(prediction['y'])
  #   width = int(prediction['width'])
  #   height = int(prediction['height'])

  #       # Calculate bounding box coordinates
  #   top_left = (x, y)
  #   bottom_right = (x + width, y + height)

  #       # Draw bounding box rectangle
  #   draw.rectangle([top_left, bottom_right], outline='green')

  #   # Display or save the image with bounding boxes
  #   # if output_path:
  # print('before save')
  # image_data.save(output_path)
    # else:
        # image.show()

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'url'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)

    

    # app.use_static_for('static')