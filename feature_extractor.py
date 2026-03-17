"""
MRI 特征提取器 V2（支持文件对象和路径）
- 与训练代码完全一致的特征提取流程
- 可接受 MRI 文件路径或文件对象（BytesIO、Flask 上传文件等）
- 自动处理 .hdr 格式转换（仅路径模式）
- 包含哈佛-牛津图谱特征、不对称特征等
"""

import os
import shutil
import tempfile
import warnings
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pywt
from scipy import stats
from scipy.stats import skew, kurtosis
from nilearn import datasets
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
from skimage.measure import marching_cubes, mesh_surface_area

warnings.filterwarnings('ignore')


class MRIFeatureExtractorV2:
    """
    完全按照训练代码的流程提取特征，支持 RF/SVM 模型。
    新增功能：可通过 file_obj 直接传入文件对象（如 Flask 上传的文件），
    避免将文件保存到磁盘后再传入路径。
    """

    def __init__(self,
                 common_mask_path=None,
                 atlas_version='sub-maxprob-thr25-2mm'):
        """
        初始化特征提取器

        Args:
            common_mask_path: 公共掩膜路径（组间交集掩膜），用于组掩膜特征
            atlas_version: 哈佛-牛津图谱版本
        """
        self.common_mask = None
        self.mask_data = None
        self.temp_files = []          # 记录自动生成的临时文件，以便清理
        self.atlas_img = None
        self.atlas_labels = []
        self.target_regions = []

        # 加载公共掩膜
        if common_mask_path and os.path.exists(common_mask_path):
            try:
                self.common_mask = nib.load(common_mask_path)
                self.mask_data = self.common_mask.get_fdata() > 0
                print(f"✅ 加载公共掩膜: {common_mask_path}, 形状: {self.mask_data.shape}")
            except Exception as e:
                print(f"⚠️ 公共掩膜加载失败: {e}")
        else:
            print("⚠️ 未提供公共掩膜，将使用全脑掩膜")

        # 加载哈佛-牛津图谱
        print("加载哈佛-牛津图谱...")
        try:
            atlas = datasets.fetch_atlas_harvard_oxford(atlas_version)
            self.atlas_img = atlas['maps']
            self.atlas_labels = atlas['labels']

            # 定义目标脑区（与训练代码完全一致）
            self.target_regions = [
                'Left Cerebral White Matter', 'Left Cerebral Cortex', 'Left Lateral Ventricle',
                'Left Thalamus', 'Left Caudate', 'Left Putamen', 'Left Pallidum',
                'Left Hippocampus', 'Left Amygdala', 'Left Accumbens',
                'Right Cerebral White Matter', 'Right Cerebral Cortex', 'Right Lateral Ventricle',
                'Right Thalamus', 'Right Caudate', 'Right Putamen', 'Right Pallidum',
                'Right Hippocampus', 'Right Amygdala', 'Right Accumbens'
            ]

            print(f"✅ 加载哈佛-牛津图谱成功，共 {len(self.atlas_labels)} 个标签")
        except Exception as e:
            print(f"❌ 哈佛-牛津图谱加载失败: {e}")
            raise

    # ==================== 图像加载统一接口 ====================
    def _load_image(self, img_path=None, file_obj=None):
        """
        统一加载 nibabel 图像：支持路径或文件对象。
        file_obj 优先级高于 img_path。

        Args:
            img_path: 文件路径字符串
            file_obj: 文件对象（如 BytesIO, Flask FileStorage）

        Returns:
            img_nib: nibabel 图像对象
            is_temp: 布尔值，表示图像是否来自临时文件（需要清理）
        """
        if file_obj is not None:
            # 从文件对象加载
            try:
                # 将文件对象指针重置到开头
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                img_nib = nib.load(file_obj)
                return img_nib, False
            except Exception as e:
                # 如果直接加载失败（例如需要解压的 .gz 文件对象可能不被 nibabel 直接支持），
                # 则保存为临时文件后再加载
                print(f"直接从文件对象加载失败，尝试保存为临时文件: {e}")
                temp_fd, temp_path = tempfile.mkstemp(suffix='.nii.gz')
                os.close(temp_fd)
                with open(temp_path, 'wb') as f:
                    file_obj.seek(0)
                    f.write(file_obj.read())
                img_nib = nib.load(temp_path)
                return img_nib, True
        elif img_path is not None:
            # 从路径加载，可能需要格式转换
            converted_path, is_temp = self._check_and_convert_format(img_path)
            img_nib = nib.load(converted_path)
            return img_nib, is_temp
        else:
            raise ValueError("必须提供 img_path 或 file_obj")

    def _check_and_convert_format(self, img_path):
        """
        检查和转换 MRI 图像格式为 .nii.gz 格式（仅路径模式）。
        返回转换后的路径和是否为临时文件的标志。
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"文件不存在: {img_path}")

        base_name = os.path.basename(img_path)
        file_ext = os.path.splitext(img_path)[1].lower()

        print(f"检查文件格式: {base_name}, 扩展名: {file_ext}")

        if file_ext == '.hdr':
            print("检测到.hdr格式，开始转换为.nii.gz格式...")
            temp_dir = tempfile.mkdtemp()
            temp_output = os.path.join(temp_dir, os.path.basename(img_path).replace('.hdr', '.nii.gz'))
            try:
                img = nib.load(img_path)
                nib.save(img, temp_output)
                print(f"✅ 格式转换完成: {img_path} -> {temp_output}")
                self.temp_files.append(temp_output)
                return temp_output, True
            except Exception as e:
                print(f"❌ 格式转换失败: {e}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise
        elif file_ext in ['.nii', '.gz'] or img_path.endswith('.nii.gz'):
            # 已经是 NIfTI 格式
            return img_path, False
        else:
            # 尝试直接加载，如果失败则报错
            try:
                test_img = nib.load(img_path)
                print(f"✅ 文件可加载为NIfTI格式，形状: {test_img.shape}")
                return img_path, False
            except:
                raise ValueError(f"不支持的格式或不完整的文件: {img_path}")

    # ==================== 个体掩膜生成 ====================
    def _generate_individual_mask(self, img):
        """
        使用与训练时完全一致的策略自动生成个体掩膜：
        - Z-score 标准化
        - NiftiMasker 使用 whole-brain-template 策略，smoothing_fwhm=6
        返回掩膜图像对象
        """
        data = img.get_fdata()
        if data.ndim == 4:
            data = data[..., 0]
        # Z-score 标准化
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            std_data = (data - mean) / std
        else:
            std_data = data
        std_img = nib.Nifti1Image(std_data, img.affine, img.header)

        # 生成掩膜
        masker = NiftiMasker(mask_strategy='whole-brain-template', smoothing_fwhm=6)
        masker.fit(std_img)
        mask_img = masker.mask_img_
        return mask_img

    # ==================== 特征提取核心方法 ====================
    def extract_features_simple(self, img_path=None, file_obj=None, mask_path=None, age=None, gender=None):
        """
        提取简单特征（第一部分训练代码中的特征）

        Args:
            img_path: MRI 文件路径（可选）
            file_obj: MRI 文件对象（可选，优先级高于 img_path）
            mask_path: 个体掩膜路径（如不提供则自动生成）
            age: 年龄（数值）
            gender: 性别编码后的数值（需在外部编码）

        Returns:
            features: 字典，包含简单特征
        """
        features = {}

        try:
            # 加载图像
            img_nib, is_temp = self._load_image(img_path, file_obj)
            if is_temp:
                self.temp_files.append(img_nib.get_filename())

            img_data = img_nib.get_fdata()
            if img_data.ndim == 4:
                img_data = img_data[..., 0]

            # 转换为 SimpleITK 格式用于形状计算（注意坐标顺序）
            img_sitk = sitk.GetImageFromArray(np.transpose(img_data, (2, 1, 0)))

            # 获取或生成个体掩膜
            if mask_path and os.path.exists(mask_path):
                mask_nib = nib.load(mask_path)
                mask_data = mask_nib.get_fdata() > 0
                print(f"✅ 使用提供的个体掩膜")
            else:
                # 自动生成掩膜
                mask_img = self._generate_individual_mask(img_nib)
                mask_data = mask_img.get_fdata() > 0
                print(f"✅ 自动生成个体掩膜")

            # 应用掩膜
            region_voxels = img_data[mask_data]

            # ============ 1. 全局统计特征 ============
            whole_brain_mean = np.mean(img_data) if img_data.size > 0 else 0
            region_mean = np.mean(region_voxels) if len(region_voxels) > 0 else 0
            region_std = np.std(region_voxels) if len(region_voxels) > 0 else 0
            region_skew = stats.skew(region_voxels) if len(region_voxels) > 0 else 0
            region_kurt = stats.kurtosis(region_voxels) if len(region_voxels) > 0 else 0

            features['Whole_Brain_Mean'] = float(whole_brain_mean)
            features['Region_Mean'] = float(region_mean)
            features['Region_Std'] = float(region_std)
            features['Region_Skew'] = float(region_skew)
            features['Region_Kurt'] = float(region_kurt)

            # ============ 2. 形状特征 ============
            # 创建 SimpleITK 掩膜
            mask_for_sitk = np.transpose(mask_data.astype(np.uint8), (2, 1, 0))
            mask_sitk = sitk.GetImageFromArray(mask_for_sitk)
            mask_sitk.CopyInformation(img_sitk)

            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(mask_sitk)
            label = 1

            if label_shape_filter.HasLabel(label):
                volume = label_shape_filter.GetPhysicalSize(label)          # mm³
                surface_area = label_shape_filter.GetPerimeter(label)      # mm²
                sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area if surface_area > 0 else 0
                compactness = surface_area / (4 * np.pi * ((3 * volume) / (4 * np.pi)) ** (2 / 3)) if volume > 0 else 0
                ellipsoid = label_shape_filter.GetEquivalentEllipsoidDiameter(label)

                features['Shape_Volume'] = float(volume)
                features['Shape_SurfaceArea'] = float(surface_area)
                features['Shape_Sphericity'] = float(sphericity)
                features['Shape_Compactness'] = float(compactness)
                features['Shape_EllipsoidDiameter_X'] = float(ellipsoid[0])
                features['Shape_EllipsoidDiameter_Y'] = float(ellipsoid[1])
                features['Shape_EllipsoidDiameter_Z'] = float(ellipsoid[2])
            else:
                for k in ['Volume', 'SurfaceArea', 'Sphericity', 'Compactness']:
                    features[f'Shape_{k}'] = 0.0
                for d in ['X', 'Y', 'Z']:
                    features[f'Shape_EllipsoidDiameter_{d}'] = 0.0

            # ============ 3. 小波特征长度 ============
            try:
                coeffs = pywt.wavedec2(img_data, 'db1', level=2)
                def flatten_coeffs(c):
                    flat = []
                    for item in c:
                        if isinstance(item, tuple):
                            flat.extend(flatten_coeffs(item))
                        else:
                            flat.append(item.flatten())
                    return flat
                wavelet_flat = np.concatenate(flatten_coeffs(coeffs))
                features['Wavelet_Features_Length'] = float(len(wavelet_flat))
            except Exception as e:
                print(f"小波特征提取失败: {e}")
                features['Wavelet_Features_Length'] = 0.0

            # ============ 4. 临床特征 ============
            features['Clinical_Age'] = float(age) if age is not None else 0.0
            features['Clinical_M/F'] = float(gender) if gender is not None else 0.0

            nonzero_simple = sum(1 for v in features.values() if v != 0)
            print(f"简单特征提取完成，共 {len(features)} 个，非零 {nonzero_simple} 个")
            return features

        except Exception as e:
            print(f"❌ 简单特征提取失败: {e}")
            raise

    def extract_group_mask_features(self, img_data, mask_data, group_mask_data, affine):
        """提取组掩膜区域的特征（与训练代码一致）"""
        final_mask = mask_data & group_mask_data

        if np.sum(final_mask) == 0:
            return None

        roi_values = img_data[final_mask]

        # 计算体素体积
        voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        voxel_vol = np.prod(voxel_size)

        # 计算表面积
        try:
            if final_mask.ndim == 3 and np.any(final_mask):
                verts, faces, _, _ = marching_cubes(final_mask.astype(np.float32), level=0.5)
                surface_area = mesh_surface_area(verts, faces)
                scale_factor = np.sqrt(voxel_size[0] ** 2 + voxel_size[1] ** 2 + voxel_size[2] ** 2) / np.sqrt(3)
                surface_area *= scale_factor ** 2
            else:
                surface_area = 0
        except Exception:
            surface_area = 0

        return {
            'GroupMask_Mean': float(np.mean(roi_values)),
            'GroupMask_Median': float(np.median(roi_values)),
            'GroupMask_Std': float(np.std(roi_values)),
            'GroupMask_Min': float(np.min(roi_values)),
            'GroupMask_Max': float(np.max(roi_values)),
            'GroupMask_Volume': float(np.sum(final_mask) * voxel_vol),
            'GroupMask_SurfaceArea': float(surface_area),
            'GroupMask_VoxelCount': float(np.sum(final_mask))
        }

    def extract_features_from_atlas(self, img_path=None, file_obj=None, mask_path=None):
        """
        提取哈佛-牛津图谱特征（第二部分训练代码中的特征）

        Args:
            img_path: MRI 文件路径（可选）
            file_obj: MRI 文件对象（可选，优先级高于 img_path）
            mask_path: 个体掩膜路径（如不提供则自动生成）

        Returns:
            features_dict: 字典，包含图谱特征
        """
        features_dict = {}

        try:
            # 加载图像
            img_nib, is_temp = self._load_image(img_path, file_obj)
            if is_temp:
                self.temp_files.append(img_nib.get_filename())

            # 加载或生成掩膜
            if mask_path and os.path.exists(mask_path):
                mask = nib.load(mask_path)
                mask_data = mask.get_fdata() > 0
                print(f"✅ 使用提供的个体掩膜")
            else:
                mask_img = self._generate_individual_mask(img_nib)
                mask_data = mask_img.get_fdata() > 0
                print(f"✅ 自动生成个体掩膜")

            img_data = img_nib.get_fdata()
            if img_data.ndim == 4:
                img_data = img_data[..., 0]

            # 重采样组掩膜到当前图像空间
            if self.common_mask is not None:
                group_mask_resampled = resample_to_img(
                    self.common_mask, img_nib, interpolation='nearest'
                )
                group_mask_data = group_mask_resampled.get_fdata() > 0
            else:
                group_mask_data = None

            # 重采样图谱到当前图像空间
            atlas_resampled = resample_to_img(
                self.atlas_img, img_nib, interpolation='nearest'
            )
            atlas_data = atlas_resampled.get_fdata()

            # 提取组掩膜特征
            if group_mask_data is not None:
                group_features = self.extract_group_mask_features(img_data, mask_data, group_mask_data, img_nib.affine)
                if group_features:
                    for key, value in group_features.items():
                        features_dict[key] = value
                else:
                    for feat in ['Mean', 'Median', 'Std', 'Min', 'Max', 'Volume', 'SurfaceArea', 'VoxelCount']:
                        features_dict[f'GroupMask_{feat}'] = 0.0
            else:
                for feat in ['Mean', 'Median', 'Std', 'Min', 'Max', 'Volume', 'SurfaceArea', 'VoxelCount']:
                    features_dict[f'GroupMask_{feat}'] = 0.0

            # 获取体素体积(mm³)
            voxel_size = np.sqrt(np.sum(img_nib.affine[:3, :3] ** 2, axis=0))
            voxel_vol = np.prod(voxel_size)

            # 对每个目标脑区提取特征
            total_target_regions = 0
            nonempty_regions = 0
            for i, label in enumerate(self.atlas_labels):
                if label in self.target_regions:
                    total_target_regions += 1
                    roi_mask = (atlas_data == i) & mask_data
                    region_key = label.replace(' ', '_')

                    if np.sum(roi_mask) == 0:
                        for feat in ['MeanIntensity', 'MedianIntensity', 'StdIntensity', 'MaxIntensity',
                                     'MinIntensity', 'Q25', 'Q75', 'Skewness', 'Kurtosis', 'Volume',
                                     'VoxelCount', 'CentroidX', 'CentroidY', 'CentroidZ', 'SurfaceArea',
                                     'Compactness', 'GroupMask_OverlapRatio']:
                            features_dict[f'{region_key}_{feat}'] = 0.0
                    else:
                        nonempty_regions += 1
                        roi_values = img_data[roi_mask]

                        mean_intensity = np.mean(roi_values)
                        median_intensity = np.median(roi_values)
                        std_intensity = np.std(roi_values)
                        max_intensity = np.max(roi_values)
                        min_intensity = np.min(roi_values)
                        q25 = np.percentile(roi_values, 25)
                        q75 = np.percentile(roi_values, 75)
                        skewness = skew(roi_values)
                        kurt = kurtosis(roi_values)
                        volume = np.sum(roi_mask) * voxel_vol

                        if group_mask_data is not None:
                            overlap_mask = roi_mask & group_mask_data
                            overlap_ratio = np.sum(overlap_mask) / np.sum(roi_mask) if np.sum(roi_mask) > 0 else 0
                        else:
                            overlap_ratio = 0

                        x, y, z = np.where(roi_mask)
                        centroid_vox = np.array([np.mean(x), np.mean(y), np.mean(z)])
                        centroid_mm = nib.affines.apply_affine(img_nib.affine, centroid_vox)

                        try:
                            verts, faces, _, _ = marching_cubes(roi_mask.astype(np.float32), level=0.5)
                            surface_area = mesh_surface_area(verts, faces)
                            scale = np.sqrt(voxel_size[0] ** 2 + voxel_size[1] ** 2 + voxel_size[2] ** 2) / np.sqrt(3)
                            surface_area *= scale ** 2
                        except:
                            surface_area = 0

                        compactness = (36 * np.pi * volume ** 2) / (surface_area ** 3) if surface_area > 0 else 0

                        features_dict[f'{region_key}_MeanIntensity'] = float(mean_intensity)
                        features_dict[f'{region_key}_MedianIntensity'] = float(median_intensity)
                        features_dict[f'{region_key}_StdIntensity'] = float(std_intensity)
                        features_dict[f'{region_key}_MaxIntensity'] = float(max_intensity)
                        features_dict[f'{region_key}_MinIntensity'] = float(min_intensity)
                        features_dict[f'{region_key}_Q25'] = float(q25)
                        features_dict[f'{region_key}_Q75'] = float(q75)
                        features_dict[f'{region_key}_Skewness'] = float(skewness)
                        features_dict[f'{region_key}_Kurtosis'] = float(kurt)
                        features_dict[f'{region_key}_Volume'] = float(volume)
                        features_dict[f'{region_key}_VoxelCount'] = float(np.sum(roi_mask))
                        features_dict[f'{region_key}_CentroidX'] = float(centroid_mm[0])
                        features_dict[f'{region_key}_CentroidY'] = float(centroid_mm[1])
                        features_dict[f'{region_key}_CentroidZ'] = float(centroid_mm[2])
                        features_dict[f'{region_key}_SurfaceArea'] = float(surface_area)
                        features_dict[f'{region_key}_Compactness'] = float(compactness)
                        features_dict[f'{region_key}_GroupMask_OverlapRatio'] = float(overlap_ratio)

            if total_target_regions > 0:
                print(f"图谱区域非空比例: {nonempty_regions}/{total_target_regions} ({nonempty_regions / total_target_regions:.1%})")
                if nonempty_regions / total_target_regions < 0.5:
                    print("⚠️ 警告：超过一半的图谱区域在个体掩膜内没有体素，特征将大量为0！")

            return features_dict

        except Exception as e:
            print(f"❌ 图谱特征提取失败: {e}")
            return features_dict

    def calculate_asymmetry_features(self, features_dict):
        """计算左右不对称性特征"""
        asymmetry_features = {}

        left_regions = [r for r in self.target_regions if r.startswith('Left ')]
        right_regions = [r for r in self.target_regions if r.startswith('Right ')]

        for left_region in left_regions:
            base = left_region.replace('Left ', '')
            base_clean = base.replace(' ', '_')
            right_region = f'Right {base}'

            if right_region in right_regions:
                left_vol_key = f'{left_region.replace(" ", "_")}_Volume'
                right_vol_key = f'{right_region.replace(" ", "_")}_Volume'
                left_mean_key = f'{left_region.replace(" ", "_")}_MeanIntensity'
                right_mean_key = f'{right_region.replace(" ", "_")}_MeanIntensity'
                left_surf_key = f'{left_region.replace(" ", "_")}_SurfaceArea'
                right_surf_key = f'{right_region.replace(" ", "_")}_SurfaceArea'

                left_vol = features_dict.get(left_vol_key, 0.0)
                right_vol = features_dict.get(right_vol_key, 0.0)
                left_mean = features_dict.get(left_mean_key, 0.0)
                right_mean = features_dict.get(right_mean_key, 0.0)
                left_surf = features_dict.get(left_surf_key, 0.0)
                right_surf = features_dict.get(right_surf_key, 0.0)

                vol_asym = (left_vol - right_vol) / ((left_vol + right_vol) / 2) if (left_vol + right_vol) != 0 else 0
                int_asym = (left_mean - right_mean) / ((left_mean + right_mean) / 2) if (left_mean + right_mean) != 0 else 0
                surf_asym = (left_surf - right_surf) / ((left_surf + right_surf) / 2) if (left_surf + right_surf) != 0 else 0

                asymmetry_features[f'{base_clean}_Volume_Asymmetry'] = float(vol_asym)
                asymmetry_features[f'{base_clean}_Intensity_Asymmetry'] = float(int_asym)
                asymmetry_features[f'{base_clean}_Surface_Asymmetry'] = float(surf_asym)
            else:
                base_clean = left_region.replace('Left ', '').replace(' ', '_')
                asymmetry_features[f'{base_clean}_Volume_Asymmetry'] = 0.0
                asymmetry_features[f'{base_clean}_Intensity_Asymmetry'] = 0.0
                asymmetry_features[f'{base_clean}_Surface_Asymmetry'] = 0.0

        print(f"不对称特征提取完成，共 {len(asymmetry_features)} 个")
        return asymmetry_features

    def extract_all_features(self, img_path=None, file_obj=None, subject_id="unknown",
                             mask_path=None, age=None, gender=None):
        """
        提取所有特征（合并简单特征、图谱特征、不对称特征）

        Args:
            img_path: MRI 文件路径（可选）
            file_obj: MRI 文件对象（可选，优先级高于 img_path）
            subject_id: 受试者标识（仅用于打印）
            mask_path: 个体掩膜路径（可选）
            age: 年龄（数值）
            gender: 性别编码后的数值

        Returns:
            all_features: 字典，包含所有特征
            values_list: 特征值列表（与字典顺序一致）
        """
        print(f"\n开始为{subject_id}提取特征...")

        try:
            # 1. 提取简单特征
            print("提取第一部分特征（简单特征）...")
            simple_features = self.extract_features_simple(
                img_path=img_path, file_obj=file_obj,
                mask_path=mask_path, age=age, gender=gender
            )

            # 2. 提取图谱特征
            print("提取第二部分特征（哈佛-牛津图谱特征）...")
            atlas_features = self.extract_features_from_atlas(
                img_path=img_path, file_obj=file_obj, mask_path=mask_path
            )

            # 3. 计算不对称性特征
            print("计算不对称性特征...")
            asymmetry_features = self.calculate_asymmetry_features(atlas_features)

            # 4. 合并
            all_features = {}
            all_features.update(simple_features)
            all_features.update(atlas_features)
            all_features.update(asymmetry_features)

            nonzero_total = sum(1 for v in all_features.values() if v != 0)
            print(f"✅ 特征提取完成! 共提取 {len(all_features)} 个特征，非零 {nonzero_total} 个 ({nonzero_total / len(all_features):.1%})")
            return all_features, list(all_features.values())

        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            raise

    def extract_selected_features(self, selected_features_list, img_path=None, file_obj=None,
                                  subject_id="unknown", mask_path=None, age=None, gender=None):
        """
        只提取指定的特征列表，缺失项填充0，并返回按列表顺序的值列表。
        参数同 extract_all_features，额外增加 selected_features_list。
        """
        print(f"\n开始为{subject_id}提取筛选特征...")

        all_features_dict, _ = self.extract_all_features(
            img_path=img_path, file_obj=file_obj,
            subject_id=subject_id, mask_path=mask_path,
            age=age, gender=gender
        )

        print(f"原始特征数量: {len(all_features_dict)}")
        print(f"筛选特征数量: {len(selected_features_list)}")

        filtered_feature_values = []
        missing_count = 0
        missing_samples = []

        for feat_name in selected_features_list:
            if feat_name in all_features_dict:
                filtered_feature_values.append(all_features_dict[feat_name])
            else:
                filtered_feature_values.append(0.0)
                missing_count += 1
                if len(missing_samples) < 10:
                    missing_samples.append(feat_name)

        if missing_count > 0:
            print(f"⚠️ {missing_count}个筛选特征在原始特征中未找到，已用0填充")
            if missing_samples:
                print(f"缺失特征示例: {missing_samples}")

        final_nonzero = np.count_nonzero(filtered_feature_values)
        print(f"✅ 筛选特征提取完成! 共提取 {len(filtered_feature_values)} 个特征，非零 {final_nonzero} 个 ({final_nonzero / len(filtered_feature_values):.1%})")
        return filtered_feature_values

    def cleanup_temp_files(self):
        """清理自动生成的临时文件"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    temp_dir = os.path.dirname(temp_file)
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f"✅ 清理临时目录: {temp_dir}")
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {e}")
        self.temp_files = []

    def __del__(self):
        try:
            self.cleanup_temp_files()
        except:
            pass