from concurrent.futures import process
from genericpath import exists
import os
import shutil
import sys
import glob
import argparse


from sklearn.model_selection import train_test_split
import trimesh
import numpy as np
import time


def generateSplitFiles(dataset_folder):
    class_folders = [os.path.join(dataset_folder, f) for f in os.listdir(
        dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]

    for class_f in class_folders:
        object_folders = [f for f in os.listdir(
            class_f) if os.path.isdir(os.path.join(class_f, f))]
        objects_num = len(object_folders)
        assert objects_num > 0, "None object found in {}".format(class_f)

        train_ratio = 0.6
        test_ratio = 0.2
        val_ratio = 0.2
        objs_train = objs_test = objs_val = []
        if objects_num < 5:
            if objects_num == 1:
                objs_train = objs_test = objs_val = object_folders
            elif objects_num == 2:
                objs_train = [object_folders[0]]
                objs_test = objs_val = [object_folders[1]]
            elif objects_num == 3:
                objs_train = [object_folders[0], object_folders[1]]
                objs_test = objs_val = [object_folders[2]]
            elif objects_num == 4:
                objs_train = [object_folders[0], object_folders[1]]
                objs_test = objs_val = [object_folders[2], object_folders[3]]
        else:
            objs_train, objs_test = train_test_split(
                object_folders, test_size=test_ratio, train_size=1-test_ratio, random_state=40)
            objs_train, objs_val = train_test_split(
                objs_train, test_size=val_ratio, train_size=train_ratio, random_state=41)

        # check objs number
        objs_leaked = list(set(object_folders) -
                           set(objs_train) - set(objs_test) - set(objs_val))
        if len(objs_leaked) > 0:
            objs_train += objs_leaked
        print("total {}, train set : {}, test set : {}, val set : {}".format(
            objects_num, len(objs_train), len(objs_test), len(objs_val)))

        # write train.lst
        train_lst_filename = os.path.join(class_f, 'train.lst')
        with open(train_lst_filename, 'w') as f:
            for obj in objs_train:
                f.write(obj)
                f.write('\n')
        # write test.lst
        test_lst_filename = os.path.join(class_f, 'test.lst')
        with open(test_lst_filename, 'w') as f:
            for obj in objs_test:
                f.write(obj)
                f.write('\n')
        # write val.lst
        val_lst_filename = os.path.join(class_f, 'val.lst')
        with open(val_lst_filename, 'w') as f:
            for obj in objs_val:
                f.write(obj)
                f.write('\n')


def generateWaterTightMesh(input_folder, meshfusion_ws, num_proc=0):
    if meshfusion_ws is None:
        meshfusion_ws = '../external/mesh-fusion'

    class_folders = [os.path.join(input_folder, f) for f in os.listdir(
        input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    for c_f in class_folders:
        obj_folders = [os.path.join(c_f, f) for f in os.listdir(
            c_f) if os.path.isdir(os.path.join(c_f, f))]
        for obj_f in obj_folders:
            output_folder = os.path.join(obj_f, 'temp')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            #   if os.path.exists(os.path.join(obj_f, 'for_occnet')):
            #       shutil.rmtree(os.path.join(obj_f, 'for_occnet'))

            # convert mesh to OFF
            origin_mesh_file = os.path.join(obj_f, 'mesh_watertight.ply')
            convert_mesh_folder = os.path.join(output_folder, '0_in')
            if not os.path.exists(convert_mesh_folder):
                os.makedirs(convert_mesh_folder)
            target_mesh_file = os.path.join(
                convert_mesh_folder, 'mesh_watertight.off')
            cmd = f'meshlabserver -i {origin_mesh_file} -o {target_mesh_file} '
            print(cmd)
            os.system(cmd)

            # Scaling meshes"
            scaled_mesh_folder = os.path.join(output_folder, '1_scaled')
            transed_mesh_folder = os.path.join(output_folder, '1_transform')
            cmd = f'python {meshfusion_ws}/1_scale.py --n_proc {num_proc} --in_dir {convert_mesh_folder} --out_dir {scaled_mesh_folder} --t_dir {transed_mesh_folder} '
            print(cmd)
            os.system(cmd)

            # Create depths maps
            depth_folder = os.path.join(output_folder, '2_depth')
            cmd = f'python {meshfusion_ws}/2_fusion.py --n_proc {num_proc} --mode=render --in_dir {scaled_mesh_folder} --out_dir {depth_folder} '
            print(cmd)
            os.system(cmd)

            # Produce watertight meshes
            # wt_mesh_folder = os.path.join(output_folder, '2_watertight')
            wt_mesh_folder = obj_f
            cmd = f'python {meshfusion_ws}/2_fusion.py --n_proc {num_proc} --mode=fuse --in_dir {depth_folder} --out_dir {wt_mesh_folder} --t_dir {transed_mesh_folder} '
            print(cmd)
            os.system(cmd)


def augmentMesh(input_folder):
    if not os.path.exists(input_folder):
        print(f"{input_folder} is not existing!!!")
        exit(-1)

    class_folders = [os.path.join(input_folder, class_f) for class_f in os.listdir(
        input_folder) if os.path.isdir(os.path.join(input_folder, class_f))]

    for class_folder in class_folders:
        obj_folders = [obj_f for obj_f in os.listdir(
            class_folder) if os.path.isdir(os.path.join(class_folder, obj_f))]
        for obj_f in obj_folders:
            origin_mesh_filepath = os.path.join(
                class_folder, obj_f, 'mesh_watertight.off')

            origin_mesh = trimesh.load(origin_mesh_filepath, process=False)
            mesh = origin_mesh.copy()

            for index in range(np.random.randint(low=3, high=40)):
                angle = (np.random.rand() * - 1) * np.pi
                direct = np.random.random(3)
                direct = direct / np.linalg.norm(direct)
                R = trimesh.transformations.rotation_matrix(angle, direct)
                mesh.apply_transform(R)
                dst_folder = os.path.join(class_folder, obj_f+'_'+str(index))
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                dst_mesh_filepath = os.path.join(
                    dst_folder, 'mesh_watertight.off')
                mesh.export(dst_mesh_filepath)


def main():
    parser = argparse.ArgumentParser(
        'warp igibson data into favorable format of occNet.')
    parser.add_argument('in_folder', type=str,
                        help='Path to input watertight meshes.')
    parser.add_argument('--n_proc', type=int, default=0,
                        help='Number of processes to use.')

    args = parser.parse_args()

    input_folder = args.in_folder
    num_proc = args.n_proc

    # generate watertight meshes
    # generateWaterTightMesh(input_folder, '../external/mesh-fusion', num_proc)

    # data augmentation
    # augmentMesh(input_folder)
    # print("Finish data augmentation...")

    # time.sleep(1)

    # generate split files: train.lst, test.lst, val.lst
    generateSplitFiles(input_folder)
    print("Finish generate split file...")

    # sample mesh to generate files feed into ONet
    # cmd = f'python sample_mesh.py  {input_folder} --n_proc {num_proc} --igibson --resize --packbits --float16 --overwrite'
    # print(cmd)
    # os.system(cmd)


if __name__ == "__main__":
    main()
