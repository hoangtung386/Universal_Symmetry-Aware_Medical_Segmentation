import torch
import torch.nn as nn

class ClinicalConditionEncoder(nn.Module):
    """
    Encodes clinical metadata into a dense representation for conditioning.
    Takes discrete/continuous clinical variables and outputs an embedding.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        
        # Continuous variable encoders
        self.age_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.nihss_encoder = nn.Embedding(43, 64)  # NIHSS scores 0-42

        # Categorical embeddings
        self.sex_embed = nn.Embedding(2, 32)
        self.dsa_embed = nn.Embedding(2, 32)
        
        # Feature fusion
        total_dim = 64 + 64 + 64 + 32 + 32  # Age, Time, NIHSS, Sex, DSA
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, metadata_dict):
        """
        Args:
            metadata_dict: Dictionary containing tensors for:
                'age': [B, 1] - Patient age (normalized 0-1 or raw float)
                'sex': [B] - 0=Male, 1=Female
                'nihss': [B] - Stroke severity score 0-42
                'time': [B, 1] - Time since onset (hours)
                'dsa': [B] - Digital Subtraction Angiography flag 0/1
        Returns:
            Conditioning embedding [B, embed_dim]
        """
        device = next(self.parameters()).device
        B = metadata_dict['age'].shape[0]

        # Normalize continuous variables safely
        age = metadata_dict['age'].view(B, 1).to(device) / 100.0  # Assumes max age ~100
        time = metadata_dict['time'].view(B, 1).to(device) / 24.0 # Assumes typical time < 24h
        
        # Encode continuous
        age_feat = self.age_encoder(age)
        time_feat = self.time_encoder(time)
        
        # Encode categorical
        nihss = torch.clamp(metadata_dict['nihss'].long().to(device), 0, 42)
        nihss_feat = self.nihss_encoder(nihss)
        
        sex = metadata_dict['sex'].long().to(device)
        sex_feat = self.sex_embed(sex)
        
        dsa = metadata_dict['dsa'].long().to(device)
        dsa_feat = self.dsa_embed(dsa)

        # Concatenate and fuse
        concat_feat = torch.cat([age_feat, time_feat, nihss_feat, sex_feat, dsa_feat], dim=1)
        cond_embedding = self.fusion(concat_feat)

        return cond_embedding


class ConditionalCrossAttention(nn.Module):
    """
    Injects clinical conditioning embeddings into visual features via Cross-Attention.
    """
    def __init__(self, visual_dim=1024, cond_dim=256, num_heads=8):
        super().__init__()
        
        self.num_heads = num_heads
        self.scale = (visual_dim // num_heads) ** -0.5
        
        # Query comes from Visual features
        self.q_proj = nn.Conv2d(visual_dim, visual_dim, 1)
        
        # Key/Value come from Conditioning embedding
        self.k_proj = nn.Linear(cond_dim, visual_dim)
        self.v_proj = nn.Linear(cond_dim, visual_dim)
        
        self.out_proj = nn.Conv2d(visual_dim, visual_dim, 1)
        
        self.norm1 = nn.GroupNorm(32, visual_dim)
        self.norm2 = nn.GroupNorm(32, visual_dim)
        
        # Zero initialization for residual connection (Identity mapping initially)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, visual_features, cond_embedding):
        """
        Args:
            visual_features: Tensor [B, C, H, W]
            cond_embedding: Tensor [B, cond_dim]
        Returns:
            Conditioned features [B, C, H, W]
        """
        B, C, H, W = visual_features.shape
        
        # 1. Norm visual features
        x = self.norm1(visual_features)
        
        # 2. Project visual to Queries [B, Heads, H*W, Head_Dim]
        q = self.q_proj(x).view(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)
        
        # 3. Project condition to Keys/Values [B, Heads, 1, Head_Dim]
        k = self.k_proj(cond_embedding).view(B, self.num_heads, 1, C // self.num_heads)
        v = self.v_proj(cond_embedding).view(B, self.num_heads, 1, C // self.num_heads)
        
        # 4. Cross Attention
        # Shape: [B, Heads, H*W, 1]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 5. Aggregate values
        # Shape: [B, Heads, H*W, Head_Dim] -> [B, C, H, W]
        out = (attn @ v).transpose(-1, -2).reshape(B, C, H, W)
        
        # 6. Output projection & Residual connection
        out = self.out_proj(out)
        out = visual_features + out
        
        return self.norm2(out)
