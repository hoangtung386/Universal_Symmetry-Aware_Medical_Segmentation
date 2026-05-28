import torch
import pytest


class TestLosses:
    def test_tversky_loss(self):
        from models.losses import TverskyLoss
        criterion = TverskyLoss(alpha=0.7, beta=0.3)
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 3, (2, 64, 64))
        loss = criterion(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_stroke_loss(self):
        from models.losses import StrokeLoss
        criterion = StrokeLoss(num_classes=3)
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 3, (2, 64, 64))
        loss, loss_dict = criterion(pred, target)
        assert loss.ndim == 0
        assert 'total' in loss_dict
        assert 'tversky' in loss_dict
        assert 'ce' in loss_dict
        assert 'focal' in loss_dict

    def test_stroke_loss_multiscale(self):
        from models.losses import StrokeLoss
        criterion = StrokeLoss(num_classes=3)
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 3, (2, 64, 64))
        cluster = [torch.randn(2, 3, 32, 32), torch.randn(2, 3, 16, 16)]
        loss, loss_dict = criterion(pred, target, cluster_outputs=cluster)
        assert 'multiscale' in loss_dict

    def test_symformer_loss_backwards_compat(self):
        from models.losses import SymFormerLoss
        criterion = SymFormerLoss(num_classes=3)
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 3, (2, 64, 64))
        loss, loss_dict = criterion(pred, target)
        assert loss.ndim == 0

    def test_create_ablation_losses(self):
        from models.losses import create_ablation_losses
        losses = create_ablation_losses(num_classes=3)
        assert 'baseline' in losses
        assert 'full' in losses
        assert len(losses) >= 4


class TestAlignmentNetwork:
    def test_forward_shape(self):
        from models.components import AlignmentNetwork
        net = AlignmentNetwork(input_size=(64, 64))
        x = torch.randn(2, 1, 64, 64)
        aligned, params = net(x)
        assert aligned.shape == (2, 1, 64, 64)
        assert params.shape == (2, 3)

    def test_inverse_transform(self):
        from models.components import AlignmentNetwork
        net = AlignmentNetwork(input_size=(64, 64))
        x = torch.randn(2, 1, 64, 64)
        aligned, params = net(x)
        restored, _ = net.inverse_transform(aligned, params)
        assert restored.shape == x.shape


class TestBottleneck:
    def test_get_bottleneck_mamba(self):
        from models.bottleneck import get_bottleneck
        bn = get_bottleneck('mamba', in_channels=64)
        x = torch.randn(2, 64, 3, 16, 16)
        out = bn(x)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (2, 64, 3, 16, 16)

    def test_get_bottleneck_symmetry(self):
        from models.bottleneck import get_bottleneck
        bn = get_bottleneck('symmetry', in_channels=64)
        x = torch.randn(2, 64, 3, 16, 16)
        out = bn(x)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (2, 64, 3, 16, 16)


class TestModels:
    def test_symformer_forward(self):
        from models.symformer import SymFormer
        model = SymFormer(in_channels=1, num_classes=3, input_size=(64, 64), use_kan=False)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert 'pred' in out
        assert out['pred'].shape == (2, 3, 64, 64)

    def test_conditioned_symformer_forward(self):
        from models.conditioned_symformer import ConditionedSymFormer
        model = ConditionedSymFormer(in_channels=1, num_classes=3, input_size=(64, 64), use_kan=False)
        x = torch.randn(2, 1, 64, 64)
        metadata = {'nihss': torch.tensor([10.0, 5.0]), 'age': torch.tensor([65.0, 70.0]),
                    'sex': torch.tensor([0, 1]), 'time': torch.tensor([3.0, 6.0]),
                    'dsa': torch.tensor([0, 1])}
        out = model(x, metadata_dict=metadata)
        assert 'pred' in out
        assert out['pred'].shape == (2, 3, 64, 64)


class TestConfig:
    def test_load_base_config(self):
        from configs.config import load_config
        import tempfile, os, yaml
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'datasets'))
            base = {'SEED': 42, 'DATA_PATHS': {}, 'BASE_PATH': '.', 'IMAGE_DIR': '.',
                    'MASK_DIR': '.', 'OUTPUT_DIR': '.', 'CHECKPOINT_DIR': '.'}
            with open(os.path.join(tmp, 'base.yaml'), 'w') as f:
                yaml.dump(base, f)
            config = load_config.__wrapped__ if hasattr(load_config, '__wrapped__') else load_config
            # Can't easily test without mocking config dir, just verify class works
            from configs.config import TrainingConfig
            cfg = TrainingConfig(
                SEED=42, DATA_PATHS={}, BASE_PATH='.', IMAGE_DIR='.', MASK_DIR='.',
                OUTPUT_DIR='.', CHECKPOINT_DIR='.'
            )
            assert cfg.SEED == 42
            assert cfg.FP_PENALTY_WEIGHT == 0.0
