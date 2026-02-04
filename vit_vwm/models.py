import timm
import torch
import torch.nn as nn


class ViT_control(nn.Module):
    """
    Unbounded ViT for Visual Working Memory Tasks (Control Condition)

    Architecture Overview:
    ----------------------
    This model performs the same spatial retro-cue task as BaselineViT_Spatial
    but WITHOUT the attention bottleneck that induces competition.

    Information Flow:
    -----------------
    Memory Image  →  Encoder  →  Global Pool  →  768-dim
                                                    ↓
                                              Concatenate  →  1536-dim  →  Readout  →  [sin θ, cos θ]
                                                    ↑
    Probe Image   →  Encoder  →  Global Pool  →  768-dim

    Key Properties:
    ---------------
    1. NO ATTENTION BOTTLENECK: Memory features are globally averaged, not
       competitively selected. All items contribute equally to the pooled
       representation regardless of probe location.

    2. PROBE AS CONTEXT: The probe features are concatenated with memory
       features, allowing the readout head to learn location-dependent
       decoding without explicit attention-based selection.

    3. UNBOUNDED CAPACITY: No architectural constraint forces items to
       compete for representation. Any capacity limits that emerge must
       arise from representational interference, not selection bottlenecks.
    """

    def __init__(self):
        super().__init__()

        # =============================================================
        # 1. VISUAL ENCODER (Shared for Memory and Probe)
        # =============================================================
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )

        # =============================================================
        # 2. FREEZING STRATEGY (Identical to Baseline for Fair Comparison)
        # =============================================================
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.blocks[-1].parameters():
            param.requires_grad = True

        for param in self.encoder.norm.parameters():
            param.requires_grad = True

        self.enc_dim = 768

        # =============================================================
        # 3. READOUT HEAD (Takes Concatenated Features)
        # =============================================================
        # Input: 768 (memory) + 768 (probe) = 1536
        # This head must learn to use probe information to decode the
        # relevant color from the pooled memory representation.
        self.readout = nn.Sequential(
            nn.LayerNorm(self.enc_dim * 2),
            nn.Linear(self.enc_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, mem_img, probe_img):
        """
        Args:
            mem_img:   [batch, 3, 224, 224] - Memory display with colored items
            probe_img: [batch, 3, 224, 224] - Probe display with white cue

        Returns:
            [batch, 2] - Predicted color as [sin(θ), cos(θ)]
        """
        # ---------------------------------------------------------
        # Step 1: Encode both images
        # ---------------------------------------------------------
        mem_patches = self.encoder(mem_img)      # [batch, 197, 768]
        probe_patches = self.encoder(probe_img)  # [batch, 197, 768]

        # ---------------------------------------------------------
        # Step 2: Global average pooling (NO COMPETITION)
        # ---------------------------------------------------------
        # All spatial patches contribute equally—no attention-based
        # selection mechanism to induce competition among items.
        mem_feat = mem_patches.mean(dim=1)       # [batch, 768]
        probe_feat = probe_patches.mean(dim=1)   # [batch, 768]

        # ---------------------------------------------------------
        # Step 3: Concatenate and decode
        # ---------------------------------------------------------
        # The readout head receives both memory and probe features.
        # It must learn to extract the relevant color based on the
        # probe's location information, but without explicit spatial
        # selection via attention.
        combined = torch.cat([mem_feat, probe_feat], dim=1)  # [batch, 1536]
        output = self.readout(combined)

        return output


class ViT_attention_bottleneck(nn.Module):
    """
    Attention-Bottlenecked ViT for Visual Working Memory Tasks

    Architecture Overview:
    ----------------------
    This model performs a spatial retro-cue task: given a memory display
    (colored squares) and a probe display (white cue at one location),
    report the color at the cued location.

    The key architectural feature is a SINGLE-QUERY CROSS-ATTENTION mechanism
    that creates an information bottleneck—the entire memory array must be
    compressed into one 768-dim vector via attention-weighted averaging.
    This bottleneck induces competition among memory items, hypothesized to
    produce human-like capacity limits.

    Information Flow:
    -----------------
    Memory Image  →  Encoder  →  Patch Embeddings (Keys/Values)
                                        ↓
    Probe Image   →  Encoder  →  Max Pool  →  Query Adapter  →  Query
                                        ↓
                              Cross-Attention (Softmax)
                                        ↓
                              Retrieved Vector (768-dim) ← BOTTLENECK
                                        ↓
                                    Readout
                                        ↓
                                  [sin θ, cos θ]
    """

    def __init__(self):
        super().__init__()
        # =============================================================
        # 1. VISUAL ENCODER (Shared for Memory and Probe)
        # =============================================================
        # ViT-Base/16: 12 transformer blocks, 768-dim embeddings, 16x16 patches
        # Output shape: [batch, 197, 768] = 1 CLS token + 196 spatial patches
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0,        # Remove classification head
            global_pool=''        # Return all patch tokens, not just CLS
        )

        # =============================================================
        # 2. FREEZING STRATEGY: Partial Fine-Tuning
        # =============================================================
        # Rationale: ImageNet-pretrained features are good for general vision
        # but not optimized for our sparse, geometric stimuli. We freeze early
        # layers (generic edge/texture detectors) and fine-tune only the final
        # block to adapt high-level representations to our task.

        # Freeze all parameters by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze final transformer block (Block 11, zero-indexed)
        # This allows learning task-specific spatial attention patterns
        for param in self.encoder.blocks[-1].parameters():
            param.requires_grad = True

        # Unfreeze final LayerNorm (required for stable fine-tuning)
        for param in self.encoder.norm.parameters():
            param.requires_grad = True

        self.enc_dim = 768

        # =============================================================
        # 3. PROBE QUERY ADAPTER
        # =============================================================
        # Transforms the probe's visual features into an attention query.
        # The probe image contains a white square at the target location;
        # this adapter learns to convert that visual signal into a
        # "request" for information at that spatial position.
        self.probe_adapter = nn.Sequential(
            nn.LayerNorm(self.enc_dim),
            nn.Linear(self.enc_dim, self.enc_dim),
            nn.ReLU(),
            nn.Linear(self.enc_dim, self.enc_dim)
        )

        # Scaled dot-product attention (as in Vaswani et al., 2017)
        self.attn_scale = self.enc_dim ** -0.5

        # =============================================================
        # 4. COLOR READOUT HEAD
        # =============================================================
        # Decodes the retrieved memory vector into a color prediction.
        # Output: [sin(θ), cos(θ)] unit vector representing hue angle.
        # This circular parameterization avoids discontinuity at 0°/360°
        self.readout = nn.Sequential(
            nn.LayerNorm(self.enc_dim),
            nn.Linear(self.enc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, mem_img, probe_img):
        """
        Args:
            mem_img:   [batch, 3, 224, 224] - Memory display with colored items
            probe_img: [batch, 3, 224, 224] - Probe display with white cue

        Returns:
            [batch, 2] - Predicted color as [sin(θ), cos(θ)]
        """
        # ---------------------------------------------------------
        # Step 1: Encode both images into patch representations
        # ---------------------------------------------------------
        # mem_patches:   [batch, 197, 768] - All spatial information available
        # probe_patches: [batch, 197, 768] - Contains cue location signal
        mem_patches = self.encoder(mem_img)
        probe_patches = self.encoder(probe_img)

        # ---------------------------------------------------------
        # Step 2: Extract probe query via max-pooling
        # ---------------------------------------------------------
        # Max-pool across patches to isolate the strongest signal (the white cue).
        # This is a simple way to extract "where is the cue?" without
        # explicit location detection.
        # probe_feat: [batch, 768]
        probe_feat, _ = probe_patches.max(dim=1)

        # ---------------------------------------------------------
        # Step 3: Transform probe features into attention query
        # ---------------------------------------------------------
        # query: [batch, 1, 768] - Single query vector (this is the bottleneck!)
        query = self.probe_adapter(probe_feat).unsqueeze(1)

        # ---------------------------------------------------------
        # Step 4: Cross-attention to retrieve from memory
        # ---------------------------------------------------------
        # Attention weights determine how much each memory patch contributes.
        # Softmax creates COMPETITION: weights must sum to 1, so attending
        # to one patch means attending less to others.
        #
        # attn_logits: [batch, 1, 197] - Raw similarity scores
        # attn_weights: [batch, 1, 197] - Normalized attention distribution
        attn_logits = torch.bmm(query, mem_patches.transpose(1, 2)) * self.attn_scale
        attn_weights = torch.softmax(attn_logits, dim=-1)

        # ---------------------------------------------------------
        # Step 5: Retrieve memory content via attention-weighted sum
        # ---------------------------------------------------------
        # THIS IS THE CRITICAL BOTTLENECK:
        # All 197 memory patches (potentially encoding 8 items) are compressed
        # into a SINGLE 768-dim vector. When attention is divided among multiple
        # items, this vector becomes a blended average—degrading retrieval quality.
        #
        # retrieved_memory: [batch, 768]
        retrieved_memory = torch.bmm(attn_weights, mem_patches).squeeze(1)

        # ---------------------------------------------------------
        # Step 6: Decode retrieved vector into color prediction
        # ---------------------------------------------------------
        output = self.readout(retrieved_memory)

        return output
