# machine_learning_project
this is first ML project

### software requirements
1.  [GIT Documentation](https://git-scm.com/docs/gittutorial)



conda create -p venv python==3.7 -y

conda activate venv/ --- to activate virtual environment


creating conda environment

pip install -r requirements.txt --- cmd to install flask

to add files to git ----- git add. or git add <file_name>

note : to ignore file or folder from git we can write name of file /folder in .gitignore file

to check the git status --------- git status

to check all versions maintained by git ---------- git log

to create version/comit all changes by git ---------- git commit -m 'message'

to send versions/changes to github ------- git push origin main

to check remote url ----------- git remote -v

to set up CI/CD pipeline we need 3 informations

1. HEROKU_EMAIL = ritesh.nisanka@gmail.com
2. HEROKU_API_KEY = 05ad70bc-11b7-4482-9504-ad54b5e1f34c
3. HEROKU_APP_NAME = mi-regression-app

BUILD DOCKER IMAGE

```
docker build -t <image_name>:<tagname>
```
>Note:Image name for docker must be in lower case

to list docker image
```
docker images
```

run docker image
```
docker run -p 5000:5000 -e PORT=5000 <image id>
```

to check running container 
```
docker ps
```

to stop docker container
```
docker stop <cointainer_id>
```


```
python setup.py install
```
no need to write pip install -r requirement.txt, if we write python setup.py install

install ipynbkernel

```
pip install ipykernel
```
