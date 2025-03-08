"""glb文件转obj文件
"""

import os
import cv2
import json
import torch
import trimesh
import glob
import time
from tqdm import tqdm
import concurrent.futures
import torch.multiprocessing as mp


class Mesh:
    def __init__(self, mesh_path, target_scale=1.0, mesh_dy=0.0,
                 remove_mesh_part_names=None, remove_unsupported_buffers=None, intermediate_dir=None):
        # from https://github.com/threedle/text2mesh
        self.material_cvt, self.material_num, org_mesh_path, is_convert = None, 1, mesh_path, False
        if not mesh_path.endswith(".obj") and not mesh_path.endswith(".off"):
            if mesh_path.endswith(".gltf"):
                mesh_path = self.preprocess_gltf(mesh_path, remove_mesh_part_names, remove_unsupported_buffers)
            mesh_temp = trimesh.load(mesh_path, force='mesh', process=True, maintain_order=True)

            mesh_path = os.path.splitext(mesh_path)[0] + "_cvt.obj"
            mesh_temp.export(mesh_path)
            merge_texture_path = os.path.join(os.path.dirname(mesh_path), "material_0.png")
            if os.path.exists(merge_texture_path):
                self.material_cvt = cv2.imread(merge_texture_path)
                self.material_num = self.material_cvt.shape[1] // self.material_cvt.shape[0]
            # logger.info("Converting current mesh model to obj file with {} material~".format(self.material_num))
            print("Converting current mesh model to obj file with {} material~".format(self.material_num))
            is_convert = True

        '''if ".obj" in mesh_path:
            try:
                mesh = kal.io.obj.import_mesh(mesh_path, with_normals=True, with_materials=True)
                print('loaded mesh with material')
            except:
                mesh = kal.io.obj.import_mesh(mesh_path, with_normals=True, with_materials=False)
        elif ".off" in mesh_path:
            mesh = kal.io.off.import_mesh(mesh_path)
        else:
            raise ValueError(f"{mesh_path} extension not implemented in mesh reader.")

        self.vertices = mesh.vertices.to(device)    
        self.faces = mesh.faces.to(device)          
        try:
            self.vt = mesh.uvs                          
            self.ft = mesh.face_uvs_idx   
            print('Obtained uvs from loaded mesh directly')              
        except AttributeError:
            self.vt = None
            self.ft = None
        self.mesh_path = mesh_path
        self.normalize_mesh(target_scale=target_scale, mesh_dy=mesh_dy)

        if is_convert and intermediate_dir is not None:
            if not os.path.exists(intermediate_dir):
                os.makedirs(intermediate_dir)
            if os.path.exists(os.path.splitext(org_mesh_path)[0] + "_removed.gltf"):
                os.system("mv {} {}".format(os.path.splitext(org_mesh_path)[0] + "_removed.gltf", intermediate_dir))
            if mesh_path.endswith("_cvt.obj"):
                os.system("mv {} {}".format(mesh_path, intermediate_dir))
            os.system("mv {} {}".format(os.path.join(os.path.dirname(mesh_path), "material.mtl"), intermediate_dir))
            if os.path.exists(merge_texture_path):
                os.system("mv {} {}".format(os.path.join(os.path.dirname(mesh_path), "material_0.png"), intermediate_dir))'''

    def preprocess_gltf(self, mesh_path, remove_mesh_part_names, remove_unsupported_buffers):
        with open(mesh_path, "r") as fr:
            gltf_json = json.load(fr)
            if remove_mesh_part_names is not None:
                temp_primitives = []
                for primitive in gltf_json["meshes"][0]["primitives"]:
                    if_append, material_id = True, primitive["material"]
                    material_name = gltf_json["materials"][material_id]["name"]
                    for remove_mesh_part_name in remove_mesh_part_names:
                        if material_name.find(remove_mesh_part_name) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_primitives.append(primitive)
                gltf_json["meshes"][0]["primitives"] = temp_primitives
                print("Deleting mesh with materials named '{}' from gltf model ~".format(remove_mesh_part_names))

            if remove_unsupported_buffers is not None:
                temp_buffers = []
                for buffer in gltf_json["buffers"]:
                    if_append = True
                    for unsupported_buffer in remove_unsupported_buffers:
                        if buffer["uri"].find(unsupported_buffer) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_buffers.append(buffer)
                gltf_json["buffers"] = temp_buffers
                print("Deleting unspported buffers within uri {} from gltf model ~".format(remove_unsupported_buffers))
            updated_mesh_path = os.path.splitext(mesh_path)[0] + "_removed.gltf"
            with open(updated_mesh_path, "w") as fw:
                json.dump(gltf_json, fw, indent=4)
        return updated_mesh_path

    def normalize_mesh(self, target_scale=1.0, mesh_dy=0.0):
        print('in mesh normalization, the target scale is ', target_scale)
        verts = self.vertices
        center = verts.mean(dim=0)
        print(center)
        verts = verts - center        
        
        scale = torch.max(torch.norm(verts, p=2, dim=1))   
        print('scale is ', scale)
        print('target_scale is ', target_scale)
        verts = verts *  target_scale
        
        # verts[:, 0] = verts[:, 0] + 0.03138578
        # verts[:, 1] = verts[:, 1] - 0.03138578
        # verts[:, 2] = verts[:, 2] + 0.02057767
        
        verts[:, 1] += mesh_dy   
        print('mesh_dy is ', mesh_dy)
        self.vertices = verts



# 串行执行
def main(glb_files):
    for i, glb_path in enumerate(tqdm(glb_files)):   
        tqdm.write('processing glb file {}'.format(glb_path))
        start_time = time.time()
        try:
            mesh = Mesh(glb_path)
        except:
            log_path = os.path.join('log', 'log.txt')
            with open(log_path, 'a') as f:
                f.write(f'err glb path: {glb_path} \n')

        if i == 0:
            print('time for mesh loading is ', time.time()-start_time) 


# 并行执行
def process_chunk(ins_chunk):

    for ins_path in ins_chunk:
        # 将执行的类别写到log.txt中
        log_path = os.path.join('log', 'log.txt')
        with open(log_path, 'a') as f:
            f.write(f'ins glb path: {ins_path} \n')
        mesh = Mesh(ins_path)
def main_parallel(glb_files):


    # 将类别分块，每块由一个GPU处理
    chunk_size = 300  # 越大，进程数越少
    ins_chunks = [glb_files[i:i+chunk_size] for i in range(0, len(glb_files), chunk_size)]
    # if 

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
    #     futures = [executor.submit(process_chunk, data_type_root, save_type_root, data_type, semantic_label_path, gpu_id, cate_chunks[gpu_id]) for gpu_id in range(num_gpus)]
    #     for future in concurrent.futures.as_completed(futures):
    #         future.result()


if __name__=="__main__":


    # 串行
    glbs_root = 'F:/2024_11_06_13_50_251730872230.678803/'
    glb_files = glob.glob(os.path.join(glbs_root, '*.glb'), recursive=True)
    # glb_files = ['/data4/jl/datasets/objaverse_LVIS/rearranged_data/power_shovel/ins_012/2d671887dbb845aea4edc4ea23702874.glb']
    main(glb_files)

    # # 并行
    # mp.set_start_method('spawn') # 主进程调用
    # glbs_root = '/data4/jl/datasets/objaverse_LVIS/rearranged_data/'
    # glb_files = glob.glob(os.path.join(glbs_root, '**/*.glb'), recursive=True)
    # main_parallel(glb_files)
    
    