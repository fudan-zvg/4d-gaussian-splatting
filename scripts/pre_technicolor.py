# Adapted from https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/script/pre_technicolor.py

import os 
import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle

import natsort 
import struct
import pickle
import csv
import sys 
import argparse
from PIL import Image
import sqlite3

sys.path.append(".")

# source: https://github.com/colmap/colmap/blob/dev/scripts/python/database.py
IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),
             array_to_blob(qvec), array_to_blob(tvec)))


def getcolmapsingletechni(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder + " --SiftExtraction.use_gpu 0"

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    
    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile + " --SiftMatching.use_gpu 0"
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  #
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)


    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
    
    return 


def convertmodel2dbfiles(path, offset=0):
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    manualfolder = os.path.join(projectfolder, "manual")


    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))

    db.create_tables()


    with open(os.path.join(path, "cameras_parameters.txt"), "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                idx = idx - 1
                row = [float(c) for c in row if c.strip() != '']
                fx = row[0]  
                fy = row[0]  

                cx = row[1]  
                cy = row[2]  

                colmapQ = [row[5], row[6], row[7], row[8]] 
                colmapT = [row[9], row[10], row[11]]  
                cameraname = "cam" + str(idx).zfill(2)
                focolength = fx

                principlepoint =[0,0]
                principlepoint[0] = cx 
                principlepoint[1] = cy  
                 
                imageid = str(idx+1)
                cameraid = imageid
                pngname = cameraname + ".png"

                line =  imageid + " "

                for j in range(4):
                    line += str(colmapQ[j]) + " "
                for j in range(3):
                    line += str(colmapT[j]) + " "
                line = line  + cameraid + " " + pngname + "\n"
                empltyline = "\n"
                imagetxtlist.append(line)
                imagetxtlist.append(empltyline)

                newwidth = 2048
                newheight = 1088
                params = np.array((fx , fy, cx, cy,))

                camera_id = db.add_camera(1, newwidth, newheight, params)     # RADIAL_FISHEYE                                                                                 # width and height

                cameraline = str(idx+1) + " " + "PINHOLE " + str(newwidth) +  " " + str(newheight) + " " + str(focolength) + " " + str(focolength)  + " " + str(cx) + " " + str(cy)  + "\n"
                cameratxtlist.append(cameraline)
                image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((colmapT[0], colmapT[1], colmapT[2])), image_id=idx+1)
                db.commit()
                print("commited one")
    db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 


def imagecopy(video, offsetlist=[0],focalscale=1.0, fixfocal=None):
    import cv2
    import numpy as np
    import os 
    import json 
    
    pnglist = glob.glob(video + "/*.png")

    for pngpath in pnglist:
        pass 
    
    for idx , offset in enumerate(offsetlist):
        pnglist = glob.glob(video + "*_undist_" + str(offset).zfill(5)+"_*.png")
        
        targetfolder = os.path.join(video, "colmap_" + str(idx), "input")
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for pngpath in pnglist:
            cameraname = os.path.basename(pngpath).split("_")[3]
            newpath = os.path.join(targetfolder, "cam" + cameraname )
            shutil.copy(pngpath, newpath)
    

def checkimage(videopath):
    from PIL import Image

    import cv2
    imagelist = glob.glob(videopath + "*.png")
    for imagepath in imagelist:
        try:
            img = Image.open(imagepath) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
                print('Bad file:', imagepath) # print out the names of corrupt files
        bad_file_list=[]
        bad_count=0
        try:
            img.cv2.imread(imagepath)
            shape=img.shape # this will throw an error if the img is not read correctly
        except:
            bad_file_list.append(imagepath)
            bad_count +=1
    print(bad_file_list)

def fixbroken(imagepath, refimagepath):
    try:
        img = Image.open(imagepath) # open the image file
        print("start verifying", imagepath)
        img.verify() # if we already fixed it. 
        print("already fixed", imagepath)
    except :
        print('Bad file:', imagepath)
        import cv2
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(imagepath)
        
        img.load()
        img.save("tmp.png")

        savedimage = cv2.imread("tmp.png")
        mask = savedimage == 0
        refimage = cv2.imread(refimagepath)
        composed = savedimage * (1-mask) + refimage * (mask)
        cv2.imwrite(imagepath, composed)
        print("fixing done", imagepath)
        os.remove("tmp.png")


if __name__ == "__main__" :
    scenenamelist = ["Train"]
    framerangedict = {}
    framerangedict["Birthday"] = [_ for _ in range(151, 201)] # start from 1
    framerangedict["Fabien"] = [_ for _ in range(51, 101)] # start from 1
    framerangedict["Painter"] = [_ for _ in range(100, 150)] # start from 0
    framerangedict["Theater"] = [_ for _ in range(51, 101)] # start from 1
    framerangedict["Train"] = [_ for _ in range(151, 201)] # start from 1
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--videopath", default="", type=str)
    args = parser.parse_args()

    videopath = args.videopath

    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    srcscene = videopath.split("/")[-2]
    print("srcscene", srcscene)

    if srcscene == "Birthday":
        print("check broken")
        fixbroken(videopath + "Birthday_undist_00173_09.png", videopath + "Birthday_undist_00172_09.png")
        
    imagecopy(videopath, offsetlist=framerangedict[srcscene])

    for offset in tqdm.tqdm(range(0, 50)):
        convertmodel2dbfiles(videopath, offset=offset)

    for offset in range(0, 50):
        getcolmapsingletechni(videopath, offset=offset)



