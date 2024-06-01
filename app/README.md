# Web tool "ROC Analysis"

This project was created to perform a user-friendly web tool for ROC analysis by Yury Kirpichev @ Moscow Center of 
Diagnostics and Telemedicine www.mosmed.ai

It is an open-source project under [Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/)

If you use this project or any part of it, please don't forget to mention us. 

To perform the ROC analysis just drag-n-drop and excel or csv format into the drag-n-drop place on the web page.
The file must:
- be in '.xlsx' or '.csv' format
- have an any column to identify each study
- have "GT" column for true (ground truth) value
- have "result" column for predicted values

## Building the web-site:
The web-site could be created in to ways via a Python and via Docker
### Running via Python:
Required: 
- Python v3.9 
- libraries from app_requirements.txt

Run the script "app.py"
The web-site will be available at http://0.0.0.0:8080/

### Running via Docker
To run via Docker it is required to install Docker
More info how to do it could be found at https://www.docker.com

Building the docker project:

    docker-compose build
to start the project:

    docker-compose up
  
The web-site will be available at http://0.0.0.0:8080/
