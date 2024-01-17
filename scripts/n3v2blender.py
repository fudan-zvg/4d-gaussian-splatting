import os
import argparse
import glob

import numpy as np
import json
import sys
import math
import shutil
import sqlite3

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

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def camTodatabase(txtfile, database_path):
    import os
    import argparse

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}

    if os.path.exists(database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # TODO: refine it.
    parser.add_argument("path", default="", help="input path to the video")
    args = parser.parse_args()

    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
        
    # extract images
    videos = [os.path.join(args.path, vname) for vname in os.listdir(args.path) if vname.endswith(".mp4")]
    images_path = os.path.join(args.path, "images/")
    os.makedirs(images_path, exist_ok=True)
    
    for video in videos:
        cam_name = video.split('/')[-1].split('.')[-2]
        do_system(f"ffmpeg -i {video} -start_number 0 {images_path}/{cam_name}_%04d.png")
        
    # load data
    images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    cams = sorted(set([im[7:12] for im in images]))
    
    poses_bounds = np.load(os.path.join(args.path, 'poses_bounds.npy'))
    N = poses_bounds.shape[0]

    print(f'[INFO] loaded {len(images)} images from {len(cams)} videos, {N} poses_bounds as {poses_bounds.shape}')

    assert N == len(cams)

    poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
    bounds = poses_bounds[:, -2:] # (N, 2)

    H, W, fl = poses[0, :, -1] 

    print(f'[INFO] H = {H}, W = {W}, fl = {fl}')

    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)

    # to homogeneous 
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

    # the following stuff are from colmap2nerf... 
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :] # swap y and z
    poses[:, 2, :] *= -1 # flip whole world upside down

    up = poses[:, 0:3, 1].sum(0)
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses = R @ poses

    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for i in range(N):
        mf = poses[i, :3, :]
        for j in range(i + 1, N):
            mg = poses[j, :3, :]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            #print(i, j, p, w)
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(f'[INFO] totp = {totp}')
    poses[:, :3, 3] -= totp

    avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()

    poses[:, :3, 3] *= 4.0 / avglen

    print(f'[INFO] average radius = {avglen}')
    
    train_frames = []
    test_frames = []
    for i in range(N):
        cam_frames = [{'file_path': im.lstrip("/").split('.')[0], 
                       'transform_matrix': poses[i].tolist(),
                       'time': int(im.lstrip("/").split('.')[0][-4:]) / 30.} for im in images if cams[i] in im]
        if i == 0:
            test_frames += cam_frames
        else:
            train_frames += cam_frames

    train_transforms = {
        'w': W,
        'h': H,
        'fl_x': fl,
        'fl_y': fl,
        'cx': W // 2,
        'cy': H // 2,
        'frames': train_frames,
    }
    test_transforms = {
        'w': W,
        'h': H,
        'fl_x': fl,
        'fl_y': fl,
        'cx': W // 2,
        'cy': H // 2,
        'frames': test_frames,
    }

    train_output_path = os.path.join(args.path, 'transforms_train.json')
    test_output_path = os.path.join(args.path, 'transforms_test.json')
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
    
    colmap_workspace = os.path.join(args.path, 'tmp')
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    W, H, cx, cy, fx, fy = int(W), int(H), train_transforms['cx'], train_transforms['cy'], train_transforms['fl_x'], train_transforms['fl_y']
    os.makedirs(os.path.join(colmap_workspace, 'created', 'sparse'), exist_ok=True)
    
    fname2pose = {}
    with open(os.path.join(colmap_workspace, 'created/sparse/cameras.txt'), 'w') as f:
        f.write(f'1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}')
        for frame in train_frames:
            if frame['time'] == 0:
                fname = frame['file_path'].split('/')[-1] + '.png'
                pose = np.array(frame['transform_matrix']) @ blender2opencv
                fname2pose.update({fname: pose})
                
    os.makedirs(os.path.join(colmap_workspace, 'images'), exist_ok=True)
    for fname in fname2pose.keys():
        os.symlink(os.path.abspath(os.path.join(images_path, fname)), os.path.join(colmap_workspace, 'images', fname))
                
    with open(os.path.join(colmap_workspace, 'created/sparse/images.txt'), 'w') as f:
        idx = 1
        for fname in fname2pose.keys():
            pose = fname2pose[fname]
            R = np.linalg.inv(pose[:3, :3])
            T = -np.matmul(R, pose[:3, 3])
            q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

            f.write(f'{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n')
            idx += 1
    
    with open(os.path.join(colmap_workspace, 'created/sparse/points3D.txt'), 'w') as f:
        f.write('')
    
    db_path = os.path.join(colmap_workspace, 'database.db')
    
    do_system(f"colmap feature_extractor \
                --database_path {db_path} \
                --image_path {os.path.join(colmap_workspace, 'images')}")
    
    camTodatabase(os.path.join(colmap_workspace, 'created/sparse/cameras.txt'), db_path)
    
    do_system(f"colmap exhaustive_matcher  \
                --database_path {db_path}")
    
    os.makedirs(os.path.join(colmap_workspace, 'triangulated', 'sparse'), exist_ok=True)
    
    do_system(f"colmap point_triangulator   \
                --database_path {db_path} \
                --image_path {os.path.join(colmap_workspace, 'images')} \
                --input_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                --output_path  {os.path.join(colmap_workspace, 'triangulated/sparse')}")
    
    do_system(f"colmap model_converter \
                --input_path  {os.path.join(colmap_workspace, 'triangulated/sparse')} \
                --output_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                --output_type TXT")
    
    os.makedirs(os.path.join(colmap_workspace, 'dense'), exist_ok=True)
    
    do_system(f"colmap image_undistorter  \
                --image_path  {os.path.join(colmap_workspace, 'images')} \
                --input_path  {os.path.join(colmap_workspace, 'created/sparse')} \
                --output_path  {os.path.join(colmap_workspace, 'dense')}")
    
    do_system(f"colmap patch_match_stereo   \
                --workspace_path   {os.path.join(colmap_workspace, 'dense')}")
    
    do_system(f"colmap stereo_fusion    \
                --workspace_path {os.path.join(colmap_workspace, 'dense')} \
                --output_path {os.path.join(args.path, 'points3d.ply')}")
    
    shutil.rmtree(colmap_workspace)
    os.remove(os.path.join(args.path, 'points3d.ply.vis'))
    
    print(f"[INFO] Initial point cloud is saved in {os.path.join(args.path, 'points3d.ply')}.")