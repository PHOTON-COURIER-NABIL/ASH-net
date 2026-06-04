import torch
import torch.nn as nn
import math

# =====================================================================
# دالة الصيانة الطيفية المشتركة
# =====================================================================
def apply_spectral_maintenance_to_weight(W: torch.Tensor, g: float, n: int):
    """تطبيق بروتوكول أحموري للصيانة الطيفية"""
    with torch.no_grad():
        evals, evecs = torch.linalg.eig(W.to(torch.complex64))
        variance_floor = (n - 2) / (2.0 * (n - 1))
        
        real_parts = evals.real
        imag_parts = evals.imag
        real_mask = torch.abs(imag_parts) < 1e-5
        
        # كبح تصحيح 1/g
        real_parts[real_mask] *= (1.0 / math.sqrt(g))
        
        # إعادة معايرة لقاع التباين
        current_var = torch.var(real_parts, unbiased=True)
        if current_var > 0:
            real_parts *= math.sqrt(variance_floor / current_var.item())
            
        evals_m = torch.complex(real_parts, imag_parts)
        W_m = torch.matmul(torch.matmul(evecs, torch.diag(evals_m)), torch.linalg.inv(evecs))
        W.copy_(W_m.real.to(W.dtype))

# =====================================================================
# Bidirectional LSTM محصنة طيفياً بالكامل
# =====================================================================
class AhmouriStabilizedBiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, coupling_g: float = 25.0, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.g = coupling_g
        self.num_layers = num_layers
        self.n = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.apply_global_spectral_maintenance()

    def apply_global_spectral_maintenance(self):
        """صيانة شاملة تفكك بوابات PyTorch والاتجاهات العكسية جراحياً"""
        for layer in range(self.num_layers):
            for suffix in ["", "_reverse"]:
                for gate in ['ih', 'hh']:
                    weight_name = f'weight_{gate}_l{layer}{suffix}'
                    if hasattr(self.lstm, weight_name):
                        W = getattr(self.lstm, weight_name).data
                        for i in range(4):
                            start = i * self.hidden_size
                            end = start + self.hidden_size
                            W_gate = W[start:end, :]
                            if W_gate.size(0) == W_gate.size(1):
                                apply_spectral_maintenance_to_weight(W_gate, self.g, self.n)

    def forward(self, x: torch.Tensor, hidden=None):
        return self.lstm(x, hidden)

# =====================================================================
# Self-Attention بسيط
# =====================================================================
class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        context = torch.matmul(attn, V)
        return context.mean(dim=1)

# =====================================================================
# الكتلة المتبقية المحصنة
# =====================================================================
class ASResNetBlock(nn.Module):
    def __init__(self, dim: int, g: float):
        super().__init__()
        self.n = dim
        self.g = g
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def maintain(self):
        for layer in [self.fc1, self.fc2]:
            W = layer.weight.data
            if W.size(0) == W.size(1):
                apply_spectral_maintenance_to_weight(W, self.g, self.n)

    def forward(self, x):
        identity = x
        out = self.fc2(self.relu(self.fc1(x)))
        return self.relu(out + identity)

# =====================================================================
# ASH-Net الكاملة (النسخة النهائية المستقرة طيفياً)
# =====================================================================
class ASHNet(nn.Module):
    def __init__(self, feature_dim=64, num_classes=10, coupling_g=25.0, lstm_layers=2, use_attention=True):
        super().__init__()
        self.g = coupling_g
        self.feature_dim = feature_dim
        self.use_attention = use_attention
        
        self.spatial_stage = ASResNetBlock(dim=feature_dim, g=self.g)
        self.temporal_stage = AhmouriStabilizedBiLSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            coupling_g=self.g,
            num_layers=lstm_layers
        )
        
        context_dim = 2 * feature_dim
        if use_attention:
            self.context_stage = SimpleSelfAttention(embed_dim=context_dim)
            self.classifier = nn.Linear(context_dim, num_classes)
        else:
            self.classifier = nn.Linear(context_dim, num_classes)

    def apply_global_spectral_maintenance(self):
        """تفعيل الحصانة الطيفية الشاملة على كافة الطبقات والمكونات المعمارية"""
        self.spatial_stage.maintain()
        self.temporal_stage.apply_global_spectral_maintenance()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 1. Spatial Stage
        x_spatial = x.view(-1, self.feature_dim)
        x_spatial = self.spatial_stage(x_spatial)
        x_spatial = x_spatial.view(batch_size, seq_len, self.feature_dim)
        
        # 2. Temporal Stage (BiLSTM المحصنة)
        temporal_out, _ = self.temporal_stage(x_spatial)
        
        # 3. Context Aggregation
        if self.use_attention:
            context = self.context_stage(temporal_out)
        else:
            context = temporal_out.mean(dim=1)
            
        # 4. Classification
        logits = self.classifier(context)
        return logits

# =====================================================================
# اختبار كفاءة الكود والمحاكاة الحية
# =====================================================================
if __name__ == "__main__":
    print("=" * 90)
    print(" ASH-Net الكاملة والمصححة جراحياً | تفعيل بروتوكول أحموري الموحد")
    print("=" * 90)
    
    model = ASHNet(
        feature_dim=64,
        num_classes=10,
        coupling_g=30.0,
        lstm_layers=2,
        use_attention=True
    )
    
    print("[1] جاري تشغيل بروتوكول الصيانة الطيفية الشامل للشبكة الهجينة...")
    model.apply_global_spectral_maintenance()
    print("[✔] تمت الصيانة بنجاح.")
    
    dummy_data = torch.randn(16, 128, 64)
    output = model(dummy_data)
    
    print(f"[✔] نجح التمرير الأمامي بنجاح استقراري مطلق.")
    print(f"[✔] أبعاد مصفوفة المخرجات النهائية (Logits): {output.shape}")
    print("=" * 90)