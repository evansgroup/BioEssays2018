
# Biomedical Image Processing with Containers and Deep Learning: An Automated Analysis Pipeline #

This git repository functions as example code to demonstrate the concepts defined into the paper with the above-mentioned title. 

We have generated a sample problem of finding circles in images. 
* The script generate_data.py will generate the phantom images that are going to be used for the sample problem. 
* The script pipeline.py analyzes the images to detect the circles on such images, stores their coordinates into a csv file and generates quality control images. 
* The Jupyter notebook analyzed_data.ipynb reads the csv file and plots histograms of the generated datasets.

We are going to execute all scripts using Docker, to isolate the configuration of the project from your base system. Towards that end, you should have Docker installed on your computer. You can find instructions on how to install Docker on Mac or Windows at [Docker Desktop](https://www.docker.com/products/docker-desktop). For Linux installation you can use the packages provided by your distribution or use the script file under installation_scripts/docker-install_linux.sh


## Docker processing demo ##

Once Docker has been installed, let's generate the container required for the image processing pipeline
`docker build -t bioessays2018/processing Processing_Image/`

Once the Docker system has finished downloading the images and installing the requirements.txt for the python system, we are able to process and generate the data:

`docker run --rm -v $(PWD):/usr/src/app bioessays2018/processing python generate_data.py`

After this, we can process the data using the pipeline:
`docker run --rm -v $(PWD):/usr/src/app bioessays2018/processing python pipeline.py`


And now we can look at the results:
`docker run -it -p 8888:8888 -v $(PWD):/home/jovyan jupyter/scipy-notebook`

After the latest command is run, we can access the notebook from a web browser within the same computer by going to the url
`http://127.0.0.1:8888/?token=<token_you_are_assigned>`

And analyze the data by using the notebook analyze_data.ipynb


# Principles used in this demo #

## Step one - stablish a data structure and stick to it
For the proof of concept we are going to count the average intensity in circles in the image. The images will be automatically generated.
In this case we are going to simulate the data and store it in the format data/acquisitions/date/subject/experiment/image.png


## Step two - generate a processing pipeline ##
We have set it up into the file pipeline.py . It performs three operations:
* Loop through the acquired images and detect circles on them if they are not previously detected
* Loop through the images and generate quality control images if they have not yet been generated
* Aggregate all circles from all images into a single CSV file

## Step three - inspect the data and draw conclusions ##
We have done it through the jupyter notebook. In this case we can see that ...


## Step Four ##
If the images are regularly acquired, then set up automated processing. This can be done on linux systems by using the cron system. Cron reads a file named crontab, which tells him what to do at what times. Such file is editted by invoking the following command: `sudo crontab -e`

Add the following line - the code will be run at 2 am every day 
`# m h  dom mon dow   command
0 2 * * * docker run --rm -v $(PWD):/usr/src/app bioessays2018/processing python pipeline.py`


## Step five (advanced) - run in a cluster - kubernetes ##
Indications are given into the installation_scripts folder.








## Note on windows shared folders
Unfortunately as of April 2018, Docker for windows does not allow to map windows mapped drives in the container. There is a workaround for this, by mounting the filesystem within the container. Such can be done by
a) running the containers with priviledged rights by adding the flight --privileged after the run command
b) mounting the share
mount -t cifs //ip_of_server/shared_name/ path_where_to_be_mounted -o usernaeme=USERNAME,password=PASSWORD





