import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avg pooling, classification layer
    
    def forward(self, x):
        return self.backbone(x)  # shape: (batch_size, 512, 7, 7)


class AttentionModule(nn.Module):
    def __init__(self, feature_dim, num_heads): 
        super(AttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.key_proj = nn.Linear(512, feature_dim) 
        self.value_proj = nn.Linear(512, feature_dim)
        
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads = self.num_heads)
        self.self_attention = nn.MultiheadAttention(feature_dim, num_heads = self.num_heads)

        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
    
    def forward(self, feature_map, queries):

        cross_keys = self.key_proj(feature_map)  # shape: (49, batch_size, feature_dim)
        cross_values = self.value_proj(feature_map)  # shape: (49, batch_size, feature_dim)

        cross_attn_output, _ = self.cross_attention(queries, cross_keys, cross_values)  # shape: (17, batch_size, feature_dim)
        cross_attn_output = self.layer_norm1(cross_attn_output + queries)
        self_attn_output, _ = self.self_attention(cross_attn_output, cross_attn_output, cross_attn_output)  # shape: (17, batch_size, feature_dim)
        self_attn_output = self.layer_norm2(self_attn_output + cross_attn_output)

        return self_attn_output
    

class KeypointAttention(nn.Module):
    def __init__(self, backbone=Backbone(), feature_dim=256, num_queries=17, num_attention_layers=6, num_heads=4):
        super(KeypointAttention, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.query_embed = nn.Parameter(torch.randn(num_queries, feature_dim)) # query shape: (17, feature_dim), learnable
        self.attention_layers = nn.ModuleList([AttentionModule(feature_dim, num_heads) for _ in range(num_attention_layers)])
        self.final_fc = nn.Linear(feature_dim, 3)  # (x, y, confidence) for each keypoint
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        feature_map = self.backbone(x)  # Backbone module
        b, c, h, w = feature_map.size()  # feature_map shape: (batch_size, 512, 7, 7)
        feature_map = feature_map.view(b, c, h * w).permute(2, 0, 1)  # Shape: (49, batch_size, 512)
        query = self.query_embed

        query = query.unsqueeze(1).expand(-1, b, -1) # shape: (17, batch_size, feature_dim)

        for attn_layer in self.attention_layers:
            query = attn_layer(feature_map, query)
        
        query = query.permute(1, 0, 2)  # shape: (batch_size, 17, feature_dim)
        output = self.final_fc(query)  # ahape: (batch_size, 17, 3)

        reg, cls = torch.split(output, [2, 1], dim=-1)
        cls = self.sigmoid(cls)
        output = torch.cat((reg, cls), dim=-1)

        return output
    

"""
batch_size = 200
input_image = torch.randn(batch_size, 3, 224, 224)  # example input

backbone = Backbone()
model = KeypointAttention(backbone, feature_dim = 256, num_queries = 17, num_attention_layers = 6, num_heads = 4)
output = model(input_image)  # Output shape: (batch_size, num_queries, 3)
print(output.shape)
"""
