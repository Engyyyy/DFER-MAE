import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block


class MaeEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                embed_dim=1024, depth=24, num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    

def mae_encoder_base_patch16_dec512d8b(**kwargs):
    model = MaeEncoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, **kwargs)
    return model


def mae_encoder_large_patch16_dec512d8b(**kwargs):
    model = MaeEncoder(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, **kwargs)
    return model


def mae_encoder_huge_patch14_dec512d8b(**kwargs):
    model = MaeEncoder(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, **kwargs)
    return model

    
# set recommended archs
mae_encoder_base_patch16 = mae_encoder_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_encoder_large_patch16 = mae_encoder_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_encoder_huge_patch14 = mae_encoder_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks