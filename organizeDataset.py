import shutil
import pandas as pd

dataset = pd.read_csv('Data/HAM10000_metadata.csv')

akiec = []
bcc = []
bkl = []
df = []
mel = []
nv = []
vasc = []

for s in range(0, len(dataset)):
    n = str(dataset.dx[s])
    imgid = str(dataset.image_id[s]) + '.jpg'
    if  n == 'akiec':
        akiec.append(imgid)
    elif n == 'bcc':
        bcc.append(imgid)
    elif n == 'bkl':
        bkl.append(imgid)
    elif n == 'df':
        df.append(imgid)
    elif n == 'mel':
        mel.append(imgid)
    elif n == 'nv':
        nv.append(imgid)
    elif n == 'vasc':
        vasc.append(imgid)
    else:
        'No diagnosis for image found'

#EDIT THE 'in X'-variable and 'Images/X/'-variable accordingly!
for item in vasc:
    shutil.move(str('Data/Images/' + item), str('Data/Images/vasc/' + item))


        