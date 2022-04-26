from .base_options import BaseOptions

class DepthOperatorOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='operator', help='render')

        parser.add_argument('--one_scene_path', type=str, default='E:/Datasets/GraspNet/TrainImages/scene_0000/kinect/', help='path to one example')
        parser.add_argument('--one_scene_path_synthetic', type=str, default='E:/Datasets/GraspNet/Synthetic/scene_0000/kinect/', help='path to one example')
        return parser