These instructions work under 
mac os x - terminal
windows - powershell


## Step zero - stablish a data structure and stick to it
For the proof of concept we are going to count the average intensity in circles in the image. The images will be automatically generated.
In this case we are going to simulate the data and store it in the format date/subject/experiment/image.png
The data will be stored in a temporal folder defined by the user

pip install virtualenv
python /Users/sierratech/.local/lib/python3.6/site-packages/virtualenv.py my_project
source my_project/bin/activate
pip install numpy
pip install scikit-image
pip install jupyter
pip install pandas


## Step one - generate your processing pipeline.
We can see the processing of an image in the following notebook.
This is often done in the desktop of the scientist with a subset of the data
We have generated pipeline.py

## Step two - generate the virtual environment where it rune
pip freeze > requirements.txt

## Step three - dockerize the environment
- Install Docker
-- If using windows - Switch Docker to use linux containers from the context menu of the docker icon
- Create a Dockerfile (set of instructions for docker so that it can create an operating system where to run the code)
- Build the image
docker build -t evanslab/python_demo  .
- You can test the environment using docker run -it pnp /bin/bash
docker run -it --rm --name bash_test  -v ${PWD}:/usr/src/myapp -w /usr/src/myapp evanslab/python_demo /bin/bash

- Now you are ready to run the pipeline anywhere with the following command
docker run -it --rm --name pipeline_test  -v ${PWD}:/usr/src/myapp -w /usr/src/myapp evanslab/python_demo python pipeline.py

- Optional - upload the image to Dockerhub
docker login
docker push evasnlab/python_demo

## Step four - set up overnight processing
- Edit the crontab file of the system. Such is done by invoking the following command:
sudo crontab -e
- Add the following line - the code will be run at 2 am every day - this runs on mac os x and linux. 
# m h  dom mon dow   command
0 2 * * * docker run -it --rm --name test  -v <path_where_code_resides>:/usr/src/myapp -w /usr/src/myapp evanslab/python_demo python pipeline.py

## Step five - inspect the data and draw conclusions
- Launch a jupyter notebook with the same configuration as before
docker run -it --rm --name jupyter_test  -p 8888:8888 -v ${PWD}:/usr/src/myapp -w /usr/src/myapp evanslab/python_demo jupyter notebook --no-browser --ip=0.0.0.0 --allow-root
- Open a browser and point it to (replace the xxxx with the token resulting from the given name)
http://127.0.0.1:8888/?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


## Step seven (advanced) - run in a cluster - kubernetes


## Note on windows shared folders
Unfortunately as of April 2018, Docker for windows does not allow to map windows mapped drives in the container. There is a workaround for this, by mounting the filesystem within the container. Such can be done by
a) running the containers with priviledged rights by adding the flight --privileged after the run command
b) mounting the share
mount -t cifs //ip_of_server/shared_name/ path_where_to_be_mounted -o usernaeme=USERNAME,password=PASSWORD





