import h5py
import numpy as np
import open3d as o3d


class Visualize_all_files():
    def __init__(self, file_path, delimiter=None, is_MVP=False, is_gt=False, index=None, background=[0, 0, 0], point_size=2.0) -> None:
        self.file_path  = file_path
        self.index      = index
        self.delimiter  = delimiter
        self.data_save  = []
        self.is_MVP     = is_MVP
        self.is_gt      = is_gt
        self.background = background
        self.point_size = point_size
    def load_data(self):
        if self.file_path[-3:]=='txt':
            assert self.delimiter is not None, "delimiter worry, remember input right delimiter"
            data_list = []
            with open(self.file_path) as f:
                data = f.readlines()
            assert self.delimiter in data[0],"delimiter is not in your date, pleacse check twice"
            for line in data:
                line = line.strip("\n")  # 去除末尾的换行符
                data_split = line.split(self.delimiter)
                temp = list(map(float, data_split))
                data_list.append(temp)
            self.data_save = np.array(data_list)
            
        elif self.file_path[-2:]=="h5":
            if not self.is_MVP:
                f = h5py.File(self.file_path)
                self.data_save = f['data'][:]                
            elif self.is_MVP:
                assert self.index is not None, "index worry, remember input right index"
                assert self.gt is not None, "gt worry, remember input right gt(True/False)"
                f = h5py.File(self.file_path, 'r')
                if not self.is_gt:
                    self.data_save = f['incomplete_pcds'][self.index]
                elif self.is_gt:
                    self.data_save = f['complete_pcds'][self.index]
            f.close()   
        

    def visualize(self):
        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(self.data_save)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=500, height=500)  # 创建窗口
        render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
        render_option.background_color = np.array([25, 25, 25])  # 设置背景色（这里为黑色）
        render_option.point_size = 4.0  # 设置渲染点的大小
        vis.add_geometry(pt)  # 添加点云
        vis.run()


if __name__ == "__main__":
    v = Visualize_all_files(r'/home/zjh/project_linux/PF-Net-Point-Fractal-Network-master/test_one/crop_ours_txt.txt', delimiter=',')
    v.visualize()