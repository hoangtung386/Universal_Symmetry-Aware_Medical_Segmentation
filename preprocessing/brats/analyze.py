"""
Complete BraTS Dataset Analyzer
Kiểm tra preprocessing, tính mean/std, và validate cấu hình
"""
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

class BraTSAnalyzer:
    """Công cụ phân tích toàn diện cho BraTS dataset"""
    
    def __init__(self, dataset_root):
        self.root = Path(dataset_root)
        self.train_dir = self.root / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        self.val_dir = self.root / "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
        
        self.modalities = ['t2f', 't1c', 't1n', 't2w']
        self.stats = {mod: {'means': [], 'stds': [], 'mins': [], 'maxs': []} 
                     for mod in self.modalities}
        
    def check_dataset_structure(self):
        """Kiểm tra cấu trúc thư mục"""
        print("="*60)
        print("KIỂM TRA CẤU TRÚC DATASET")
        print("="*60)
        
        if not self.train_dir.exists():
            print(f"❌ Train directory KHÔNG tồn tại: {self.train_dir}")
            return False
        
        subjects = list(self.train_dir.iterdir())
        print(f"✅ Train directory: {len([s for s in subjects if s.is_dir()])} subjects")
        
        if self.val_dir.exists():
            val_subjects = list(self.val_dir.iterdir())
            print(f"✅ Val directory: {len([s for s in val_subjects if s.is_dir()])} subjects")
        else:
            print(f"⚠️  Val directory không tồn tại (có thể chưa download)")
        
        return True
    
    def analyze_single_subject(self, subject_path, show_plots=False):
        """Phân tích chi tiết 1 subject"""
        subject_id = subject_path.name
        print(f"\n{'='*60}")
        print(f"Subject: {subject_id}")
        print(f"{'='*60}")
        
        results = {}
        
        for mod in self.modalities + ['seg']:
            fpath = subject_path / f"{subject_id}-{mod}.nii.gz"
            
            if not fpath.exists():
                print(f"⚠️  {mod}: File KHÔNG tồn tại")
                continue
            
            img = nib.load(fpath)
            data = img.get_fdata().astype(np.float32)
            
            if mod == 'seg':
                unique_classes = np.unique(data)
                counts = {int(c): int(np.sum(data == c)) for c in unique_classes}
                
                print(f"📊 Segmentation:")
                print(f"   Shape: {data.shape}")
                print(f"   Classes: {unique_classes}")
                print(f"   Distribution:")
                for cls, count in counts.items():
                    pct = 100 * count / data.size
                    cls_name = {0: 'Background', 1: 'NCR', 2: 'Edema', 3: 'ET'}.get(cls, f'Class-{cls}')
                    print(f"     {cls} ({cls_name}): {count:,} voxels ({pct:.2f}%)")
                
                results['seg'] = {
                    'shape': data.shape,
                    'classes': unique_classes.tolist(),
                    'distribution': counts
                }
            else:
                # Brain mask (non-zero voxels)
                brain_mask = data > 0
                brain_voxels = data[brain_mask]
                
                if len(brain_voxels) == 0:
                    print(f"⚠️  {mod}: Volume TRỐNG!")
                    continue
                
                stats = {
                    'shape': data.shape,
                    'brain_voxels': int(brain_mask.sum()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(brain_voxels.mean()),
                    'std': float(brain_voxels.std()),
                    'median': float(np.median(brain_voxels)),
                    'p01': float(np.percentile(brain_voxels, 1)),
                    'p99': float(np.percentile(brain_voxels, 99))
                }
                
                print(f"🧠 {mod.upper()}:")
                print(f"   Shape: {stats['shape']}")
                print(f"   Brain voxels: {stats['brain_voxels']:,}")
                print(f"   Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
                print(f"   Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")
                print(f"   Median: {stats['median']:.2f}")
                print(f"   P01-P99: [{stats['p01']:.1f}, {stats['p99']:.1f}]")
                
                results[mod] = stats
                
                # Store for global stats
                self.stats[mod]['means'].append(stats['mean'])
                self.stats[mod]['stds'].append(stats['std'])
                self.stats[mod]['mins'].append(stats['min'])
                self.stats[mod]['maxs'].append(stats['max'])
        
        return results
    
    def compute_global_statistics(self, num_samples=50):
        """Tính toán thống kê toàn dataset"""
        print("\n" + "="*60)
        print(f"PHÂN TÍCH TOÀN DATASET ({num_samples} samples)")
        print("="*60)
        
        subjects = sorted([s for s in self.train_dir.iterdir() if s.is_dir()])
        
        if num_samples > len(subjects):
            num_samples = len(subjects)
        
        sample_subjects = subjects[:num_samples]
        
        for subject in tqdm(sample_subjects, desc="Analyzing subjects"):
            try:
                self.analyze_single_subject(subject, show_plots=False)
            except Exception as e:
                print(f"⚠️  Lỗi khi xử lý {subject.name}: {e}")
        
        # Tổng hợp kết quả
        print("\n" + "="*60)
        print("THỐNG KÊ TỔNG HỢP")
        print("="*60)
        
        global_stats = {}
        
        for mod in self.modalities:
            if len(self.stats[mod]['means']) == 0:
                continue
            
            global_stats[mod] = {
                'mean_of_means': np.mean(self.stats[mod]['means']),
                'std_of_means': np.std(self.stats[mod]['means']),
                'mean_of_stds': np.mean(self.stats[mod]['stds']),
                'global_min': np.min(self.stats[mod]['mins']),
                'global_max': np.max(self.stats[mod]['maxs'])
            }
            
            print(f"\n{mod.upper()}:")
            print(f"  Mean (avg across volumes): {global_stats[mod]['mean_of_means']:.2f} ± {global_stats[mod]['std_of_means']:.2f}")
            print(f"  Std (avg across volumes): {global_stats[mod]['mean_of_stds']:.2f}")
            print(f"  Global range: [{global_stats[mod]['global_min']:.1f}, {global_stats[mod]['global_max']:.1f}]")
        
        return global_stats
    
    def validate_preprocessing(self, dataset_loader):
        """Kiểm tra preprocessing trong DataLoader"""
        print("\n" + "="*60)
        print("KIỂM TRA PREPROCESSING")
        print("="*60)
        
        # Lấy 1 batch
        images, masks, metadata = next(iter(dataset_loader))
        
        print(f"Batch shape:")
        print(f"  Images: {images.shape}")  # Expected: (B, 2T+1, H, W)
        print(f"  Masks: {masks.shape}")    # Expected: (B, H, W)
        
        print(f"\nImage statistics (sau preprocessing):")
        print(f"  Min: {images.min():.4f}")
        print(f"  Max: {images.max():.4f}")
        print(f"  Mean: {images.mean():.4f}")
        print(f"  Std: {images.std():.4f}")
        
        print(f"\nMask classes:")
        unique_classes = torch.unique(masks)
        print(f"  Unique values: {unique_classes.tolist()}")
        
        # Kiểm tra metadata
        print(f"\nMetadata sample:")
        for key in metadata.keys():
            print(f"  {key}: {metadata[key][0]}")
        
        return {
            'image_range': (float(images.min()), float(images.max())),
            'image_mean': float(images.mean()),
            'image_std': float(images.std()),
            'mask_classes': unique_classes.tolist()
        }
    
    def generate_normalization_config(self, global_stats):
        """Tạo file cấu hình normalization"""
        config = {
            'dataset': 'BraTS2023',
            'modalities': {},
            'recommended_preprocessing': {
                'method': 'z-score_per_volume',
                'robust_scaling': True,
                'clip_range': [-3, 3],
                'target_range': [0, 1]
            }
        }
        
        for mod, stats in global_stats.items():
            config['modalities'][mod] = {
                'global_mean': float(stats['mean_of_means']),
                'global_std': float(stats['mean_of_stds']),
                'min': float(stats['global_min']),
                'max': float(stats['global_max'])
            }
        
        output_path = self.root / 'brats_normalization_config.json'
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Đã lưu cấu hình normalization: {output_path}")
        return config
    
    def plot_intensity_distributions(self, num_samples=10):
        """Vẽ biểu đồ phân bố intensity"""
        subjects = sorted([s for s in self.train_dir.iterdir() if s.is_dir()])[:num_samples]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, mod in enumerate(self.modalities):
            all_intensities = []
            
            for subject in tqdm(subjects, desc=f"Loading {mod}"):
                fpath = subject / f"{subject.name}-{mod}.nii.gz"
                if not fpath.exists():
                    continue
                
                data = nib.load(fpath).get_fdata()
                brain_voxels = data[data > 0]
                
                # Sample để giảm memory
                if len(brain_voxels) > 10000:
                    brain_voxels = np.random.choice(brain_voxels, 10000, replace=False)
                
                all_intensities.extend(brain_voxels)
            
            if len(all_intensities) > 0:
                axes[idx].hist(all_intensities, bins=100, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{mod.upper()} Intensity Distribution')
                axes[idx].set_xlabel('Intensity')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.root / 'brats_intensity_distributions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu biểu đồ: {save_path}")
        plt.close()


def main():
    """Chạy phân tích hoàn chỉnh"""
    import sys
    
    # Tìm dataset path
    possible_paths = [
        "Dataset_BraTs",
        "../Dataset_BraTs",
        "../../Dataset_BraTs"
    ]
    
    dataset_path = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = p
            break
    
    if dataset_path is None:
        print("❌ Không tìm thấy dataset BraTS!")
        print("Vui lòng chỉ định đường dẫn:")
        print("  python analyze_brats_complete.py /path/to/Dataset_BraTs")
        sys.exit(1)
    
    print(f"📂 Dataset path: {dataset_path}\n")
    
    # Khởi tạo analyzer
    analyzer = BraTSAnalyzer(dataset_path)
    
    # 1. Kiểm tra cấu trúc
    if not analyzer.check_dataset_structure():
        sys.exit(1)
    
    # 2. Phân tích 1 subject mẫu
    subjects = list(analyzer.train_dir.iterdir())
    if subjects:
        print("\n" + "="*60)
        print("PHÂN TÍCH CHI TIẾT 1 SUBJECT MẪU")
        print("="*60)
        analyzer.analyze_single_subject(subjects[0])
    
    # 3. Thống kê toàn dataset
    global_stats = analyzer.compute_global_statistics(num_samples=50)
    
    # 4. Tạo cấu hình normalization
    config = analyzer.generate_normalization_config(global_stats)
    
    # 5. Vẽ biểu đồ
    print("\n" + "="*60)
    print("TẠO BIỂU ĐỒ PHÂN BỐ")
    print("="*60)
    analyzer.plot_intensity_distributions(num_samples=10)
    
    print("\n" + "="*60)
    print("✅ HOÀN TẤT PHÂN TÍCH")
    print("="*60)
    print(f"Kết quả đã lưu tại: {analyzer.root}")
    print("  - brats_normalization_config.json")
    print("  - brats_intensity_distributions.png")


if __name__ == "__main__":
    main()
